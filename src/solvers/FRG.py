from settings import INSTANCE_FOLDER, FRG_PATH
from solvers.solver import Solver
import subprocess
import os
from cpmp.layout import lay2file


class FRGSolver(Solver): 
    def __init__(self):
        super().__init__("FRG")

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            r = self.solve_from_layout(layout, H, max_steps)
            r = [r[0], r[1]]
            results.append(r)

        return results

    @staticmethod
    def solve_from_layout(layout, H, max_steps, return_steps=False):
        pid = os.getpid()
        filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

        try:
            lay2file(layout, filepath)

            result = subprocess.run(
                [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignment", "2"],
                check=True,
                text=True,
                capture_output=True
            )
            
            output_str = result.stdout.strip().split('\n')

            if return_steps:
                steps = []
                for line in output_str[1:]:
                    line = line.split(',')
                    steps.append((int(line[0]), int(line[1])))
                return steps
            
            output_str = output_str[0].split('\t')
            steps_str = output_str[0].strip()
            if not steps_str.isdigit():
                solved = False
                steps = float('inf')
            else:
                solved = True
                steps = int(steps_str)

            time_str = output_str[1].strip()
            return solved, steps, float(time_str)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)