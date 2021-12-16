import argparse
from brats21.slurm.batch import BatchScript, BatchProperty


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="3d_fullres")
    parser.add_argument(
        "--trainer", "-tr", type=str, default="nnUNetTrainerV2BraTSSegnet"
    )
    parser.add_argument("--task", "-t", type=str, default="Task500_Brats21")
    parser.add_argument("--fold", "-f", nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # Submit one job per fold
    for fold in args.fold:

        # Create batch script
        bscript = BatchScript(f"brats21/slurm/scripts/script_fold{fold}.sh")

        # Set batch script propertys
        job_name = f"segnet_cvfold_{fold}"
        bscript.batch_propertys["job_name"] = BatchProperty("job_name", job_name)
        bscript.batch_propertys["runtime"] = BatchProperty("time", "96:00:00")
        bscript.batch_propertys["stdout"] = BatchProperty("output", f"{job_name}.o%j")
        bscript.batch_propertys["stderr"] = BatchProperty("error", f"{job_name}.e%j")

        # Add training function call
        bscript.function_calls = [
            f"nnUNet_train {args.mode} {args.trainer} {args.task} {fold} --npz"
        ]
        bscript.write()
        bscript.run()


if __name__ == "__main__":
    main()
