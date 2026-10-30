"""Chip monitor: NPU, Intel Arc, NVIDIA usage / memory / power from real counters only.
 - usage + memory: Windows "GPU Engine" / "GPU Adapter Memory" perf counters (same source as Task Manager)
 - NVIDIA power + temperature: NVML (nvidia-ml-py), exact board power
 - Intel Arc power: N/A (no Arc-specific sensor; RAPL PP1 reads ~0 W under load, not used)
 - NPU power: no per-device counter on this laptop -> shows CPU package power (shared by CPU, NPU and Arc)."""
import re, time, logging, subprocess, collections

class ChipCounter:
    def __init__(self):
        import win32pdh; self.p = win32pdh; self.q = None; self.built = 0
        self.nv = None
        try:
            import pynvml; pynvml.nvmlInit(); self.nvml = pynvml; self.nv = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nv_name = pynvml.nvmlDeviceGetName(self.nv); self.nv_name = self.nv_name.decode() if isinstance(self.nv_name, bytes) else self.nv_name
        except Exception as e: logging.warning("NVML not available: %s", e)
        self.luids = self._find(); self._build()

    def _engines(self):
        p = self.p
        try: p.EnumObjectItems(None, None, "GPU Engine", p.PERF_DETAIL_WIZARD, 1)
        except Exception: pass
        _, inst = p.EnumObjectItems(None, None, "GPU Engine", p.PERF_DETAIL_WIZARD); types = collections.defaultdict(set)
        for i in inst:
            m = re.search(r"(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+).*engtype_(\w*)", i)
            if m: types[m.group(1)].add(m.group(2))
        return types

    def _ded(self):
        p = self.p; q = p.OpenQuery(); cs = {}
        for path in p.ExpandCounterPath(r"\GPU Adapter Memory(*)\Dedicated Usage"):
            m = re.search(r"(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)", path)
            if m: cs[m.group(1)] = p.AddCounter(q, path)
        p.CollectQueryData(q); time.sleep(0.2); p.CollectQueryData(q); out = {}
        for l, c in cs.items():
            try: out[l] = p.GetFormattedCounterValue(c, p.PDH_FMT_DOUBLE)[1] / 1048576
            except Exception: pass
        p.CloseQuery(q); return out

    def _find(self):
        types = self._engines(); res = {}
        for l, t in types.items():
            if t == {"Compute"}: res["NPU"] = l
        gpus = [l for l, t in types.items() if "3D" in t and t != {"3D"}]  # {"3D"} only = Microsoft Basic Render
        ded = self._ded()
        # Engine names identify the vendor: NVIDIA exposes OFA (optical flow), Intel exposes GSC (graphics security)
        nvl = [l for l in gpus if any(e.startswith("OFA") for e in types[l])]
        arl = [l for l in gpus if "GSC" in types[l]]
        if len(nvl) == 1 and len(arl) == 1 and nvl != arl:
            res["NVIDIA"], res["Intel Arc"] = nvl[0], arl[0]; self.mapped = True
            logging.info("chip luids %s (by engine names)", res); return res
        if self.nv is not None and len(gpus) > 1:
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.nv).used / 1048576
            gpus.sort(key=lambda l: abs(ded.get(l, 0) - used)); res["NVIDIA"] = gpus[0]; rest = gpus[1:]
        else:
            gpus.sort(key=lambda l: -ded.get(l, 0)); rest = gpus
            if len(gpus) > 1: res["NVIDIA"] = gpus[0]; rest = gpus[1:]
        # Intel Arc iGPU: the remaining 3D adapter with the least dedicated memory (the Microsoft Basic adapter has no engines)
        if rest: res["Intel Arc"] = sorted(rest, key=lambda l: ded.get(l, 0))[0]
        self.mapped = any(v > 1 for v in ded.values())  # all-zero reading -> guess; re-checked at the next rebuild
        logging.info("chip luids %s (dedicated MB %s)", res, {k: round(v) for k, v in ded.items()})
        return res

    def _build(self):
        p = self.p
        if self.q:
            try: p.CloseQuery(self.q)
            except Exception: pass
        self.q = p.OpenQuery(); self.eng = collections.defaultdict(list); self.mem = collections.defaultdict(list); self.pw = {}
        try: p.EnumObjectItems(None, None, "GPU Engine", p.PERF_DETAIL_WIZARD, 1)
        except Exception: pass
        for chip, l in self.luids.items():
            for path in p.ExpandCounterPath(r"\GPU Engine(*%s*)\Utilization Percentage" % l):
                m = re.search(r"engtype_(\w*)", path)
                try: self.eng[chip].append((m.group(1) if m else "?", p.AddCounter(self.q, path)))
                except Exception: pass
            for nm in ("Dedicated Usage", "Shared Usage"):
                for path in p.ExpandCounterPath(r"\GPU Adapter Memory(*%s*)\%s" % (l, nm)):
                    try: self.mem[chip].append((nm, p.AddCounter(self.q, path)))
                    except Exception: pass
        try:
            _, inst = p.EnumObjectItems(None, None, "Energy Meter", p.PERF_DETAIL_WIZARD)
        except Exception as e: inst = []; logging.info("no Energy Meter object: %s", e)
        for i in inst:
            if i.lower().startswith("rapl"):
                try: self.pw[i.lower()] = p.AddCounter(self.q, r"\Energy Meter(%s)\Power" % i)
                except Exception as e: logging.info("energy counter %s: %s", i, e)
        p.CollectQueryData(self.q); self.built = time.time()

    def energy_instances(self):
        return sorted(self.pw)

    def read(self):
        """-> {chip: {"use": %, "ded": MB, "shared": MB, "watts": W or None, "wnote": str, "temp": C or None}}"""
        p = self.p
        if time.time() - self.built > 3:
            if not getattr(self, 'mapped', True): self.luids = self._find()
            self._build(); time.sleep(0.1)
        p.CollectQueryData(self.q); out = {}
        def val(c):
            try: return p.GetFormattedCounterValue(c, p.PDH_FMT_DOUBLE)[1]
            except Exception: return 0.0
        rapl = {k: val(c) / 1000.0 for k, c in self.pw.items()}  # Energy Meter Power is in milliwatts
        for chip in ("NPU", "Intel Arc", "NVIDIA"):
            if chip not in self.luids: continue
            bytype = collections.defaultdict(float)
            for t, c in self.eng[chip]: bytype[t] += val(c)
            d = {"use": min(100.0, max(bytype.values()) if bytype else 0.0), "ded": 0.0, "shared": 0.0, "watts": None, "wnote": "N/A", "temp": None}
            for nm, c in self.mem[chip]: d["ded" if nm.startswith("Ded") else "shared"] += val(c) / 1048576
            if chip == "NVIDIA" and self.nv is not None:
                try:
                    n = self.nvml; d["watts"] = n.nvmlDeviceGetPowerUsage(self.nv) / 1000.0; d["wnote"] = "board power (NVML)"
                    d["temp"] = n.nvmlDeviceGetTemperature(self.nv, n.NVML_TEMPERATURE_GPU)
                    d["use"] = max(d["use"], float(n.nvmlDeviceGetUtilizationRates(self.nv).gpu))
                except Exception: pass
            # Intel Arc: no Arc-specific power sensor (RAPL PP1 reads ~0 W under load) -> N/A
            out[chip] = d
        out["_package"] = rapl.get("rapl_package0_pkg")  # whole CPU package (real RAPL counter), shown on its own line
        return out
