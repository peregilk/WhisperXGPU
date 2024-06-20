import subprocess
import argparse

def run_commands(vm_names, shard_indices=None, zone="us-central1-a"):
    if shard_indices is None:
        shard_indices = vm_names
    elif len(shard_indices) != len(vm_names):
        raise ValueError("The length of shard_indices must be the same as vm_names")

    for i, shard_index in zip(vm_names, shard_indices):
        instance_name = f"north-l4-p{i}"
        command = (f"nohup /opt/conda/bin/python /home/pere/WhisperXGPU/transcribe.py "
                   f"--shard_indices {shard_index} --batch_size 128 --output_dir output/ "
                   f"--model_name NbAiLab/nb-whisper-large --bucket nostram-transcripts2 "
                   "> /dev/null 2>&1 &")

        # Run the command on the VM
        subprocess.run([
            "gcloud", "compute", "ssh", instance_name,
            "--zone", zone,
            "--command", command
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run commands on VM instances.')
    parser.add_argument('--vm_names', nargs='+', type=int, required=True, help='List of VM numbers to run commands on.')
    parser.add_argument('--shard_indices', nargs='+', type=int, help='List of shard indices to use, must match length of vm_names.')
    parser.add_argument('--zone', type=str, default='us-central1-a', help='Zone of the VMs.')

    args = parser.parse_args()
    run_commands(args.vm_names, args.shard_indices, args.zone)
