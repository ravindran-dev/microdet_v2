from tmp.src.common_imports import *
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class TBLogger:
    def __init__(self, log_dir: Path, flush_secs: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir), flush_secs=flush_secs) if SummaryWriter else None

    def add_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag, d, step):
        if self.writer:
            self.writer.add_scalars(tag, d, step)

    def add_text(self, tag, text, step):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def add_histogram(self, tag, values, step, bins="auto"):
        if self.writer:
            self.writer.add_histogram(tag, values, step, bins=bins)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()


class CSVLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.csv_path.with_suffix(".jsonl")
        self.fieldnames = None
        self._csv_file = None
        self._csv_writer = None

        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with open(self.csv_path, "r") as f:
                r = csv.reader(f)
                try:
                    header = next(r)
                    if header:
                        self.fieldnames = header
                except StopIteration:
                    pass

        self._open_csv()

    def _open_csv(self):
        self._csv_file = open(self.csv_path, "a", newline="")
        if self.fieldnames:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)

    def _ensure_fieldnames(self, row):
        keys = list(row.keys())
        if self.fieldnames is None:
            self.fieldnames = keys
            self._csv_file.close()
            self._open_csv()
            self._csv_writer.writeheader()
        else:
            new_keys = [k for k in keys if k not in self.fieldnames]
            if new_keys:
                self.fieldnames.extend(new_keys)
                self._csv_file.close()
                existing = []
                if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                    with open(self.csv_path, "r") as f:
                        existing = list(csv.DictReader(f))
                self._csv_file = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
                self._csv_writer.writeheader()
                for r in existing:
                    self._csv_writer.writerow(r)

    def write(self, row):
        self._ensure_fieldnames(row)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        payload = dict(row)
        payload["_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.jsonl_path, "a") as jf:
            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()


class ProgressPrinter:
    def __init__(self, total: int, print_every: int = 50, prefix: str = ""):
        self.total = max(1, total)
        self.print_every = max(1, print_every)
        self.prefix = prefix

    def update(self, it: int, scalars: Optional[Dict[str, float]] = None):
        if (it + 1) % self.print_every != 0 and (it + 1) != self.total:
            return
        msg = f"{self.prefix} {it+1}/{self.total}"
        if scalars:
            msg += " | " + ", ".join(f"{k}={float(v):.4f}" for k, v in scalars.items())
        print(msg)
