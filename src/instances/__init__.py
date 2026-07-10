import os
import shutil
from settings import INSTANCE_FOLDER
from cpmp.layout import Layout
from solvers.FRG import FRGSolver
from instances.generators.instance_generator import InstanceGenerator

def generate_instances(basename: str, generator: InstanceGenerator, amount: int):
    shutil.rmtree(INSTANCE_FOLDER / basename, ignore_errors=True)
    os.makedirs(INSTANCE_FOLDER / basename)
    instances = generator.generate_instances(amount)

    for i, inst in enumerate(instances):
        filepath = INSTANCE_FOLDER / basename / f'{basename}-{i}.txt'
        with open(filepath, 'w') as f:
            f.write(f"{generator.S} {generator.N}")
            for s in inst:
                f.write("\n")
                f.write(f"{len(s)} ")
                for g in s:
                    f.write(f"{g} ")

    print("Instancias guardadas en:", INSTANCE_FOLDER / basename)

def read_instance(file, H):
    with open(INSTANCE_FOLDER / file) as f:
        S, C = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)
            
        layout = Layout(stacks,H)
    return layout

def get_instance_files(folders):
    files = []
    for folder in folders:
        folder_files = []
        for file in os.listdir(INSTANCE_FOLDER / folder):
            folder_files.append(os.path.join(folder, file))
        files.append(folder_files)
    return files

def get_cost_distribution(folder, H):
    costs = []
    solver = FRGSolver()

    for file in os.listdir(INSTANCE_FOLDER / folder):
        layout = read_instance(os.path.join(folder, file), H)
        _, cost, _ = solver.solve_from_layout(layout, H, 999999)
        costs.append(cost)

    return costs