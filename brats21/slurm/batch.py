import subprocess
from pathlib import Path
from typing import Sequence, Mapping, Optional, Union


class BatchProperty:
    """Basic class to build and set batch file propertys."""

    def __init__(self, flag: str, value: Optional[str] = None) -> None:
        self.flag = flag
        self._value = value

    def __str__(self) -> str:
        if self.value is None:
            return f"#SBATCH --{self.flag}"
        return f"#SBATCH --{self.flag}={self.value}"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: str):
        self._value = value


DEFAULT_BATCH_PROPERTYS = {
    "job_name": BatchProperty("job-name", "my_job"),
    "partition": BatchProperty("partition", "gpu"),
    "nodes": BatchProperty("nodes", "1"),
    "memory": BatchProperty("mem", "0"),
    "generic_resources": BatchProperty("gres", "gpu:1"),
    "exclusive": BatchProperty("exclusive"),
    "runtime": BatchProperty("time", "01:00:00"),
    "mail_type": BatchProperty("mail-type", "FAIL"),
    "account": BatchProperty("account", "sc-users"),
    "stdout": BatchProperty("output", "my_job.o%j"),
    "stderr": BatchProperty("error", "my_job.e%j"),
}


class BatchScript:
    """Batch script handler. Enables to write, modify and run batch scripts."""

    def __init__(
        self,
        file_name: Union[str, Path],
        batch_propertys: Mapping[str, BatchProperty] = DEFAULT_BATCH_PROPERTYS,
        function_calls: Sequence = [],
    ) -> None:
        self.file_name = Path(file_name)
        self.shebang = "#!/bin/bash"
        self.batch_propertys = batch_propertys
        self.function_calls = function_calls

    def _prepare_directories(self) -> None:
        """Create parent directories."""
        if not self.file_name.parent.is_dir():
            self.file_name.parent.mkdir(parents=True)
        if self.file_name.is_file():
            print(f"WARNING: Batch file already exists! {self.file_name}")

    def _write_batch_propertys(self, script):
        for key, property in self.batch_propertys.items():
            if not isinstance(property, BatchProperty):
                print(f"WARNING: Non BatchProperty found: {key}")
            script.writelines(str(property) + "\n")

    def _write_function_calls(self, script):
        for call in self.function_calls:
            script.writelines(str(call) + "\n")

    def write(self):
        """Write batch script to self.file_name"""
        self._prepare_directories()
        with self.file_name.open(mode="w+") as script:
            script.writelines(self.shebang + "\n")
            self._write_batch_propertys(script)
            self._write_function_calls(script)

    def run(self, command=None):
        """Run batch script in background."""

        log_file = self.file_name.with_name(f"{self.file_name.stem}_log.txt")

        if command is None:
            arg = [self.file_name]
        else:
            arg = [command, self.file_name]

        with log_file.open(mode="w") as f:
            f.writelines(
                f"RUNNING COMMAND:\n{' '.join([str(x) for x in arg])}\n\nSTDOUT:\n"
            )
            process = subprocess.Popen(arg, shell=True, stdout=f, text=True)
