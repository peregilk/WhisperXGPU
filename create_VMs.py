import subprocess
import argparse

def create_vms(vm_names, zone="us-central1-a"):
    project = "north-390910"
    machine_type = "g2-standard-4"
    service_account = "north-390910@appspot.gserviceaccount.com"
    accelerator = "count=1,type=nvidia-l4"
    image = "l4-20june2024-image"
    image_project = "north-390910"
    boot_disk_size = "200GB"
    boot_disk_type = "pd-balanced"

    for i in vm_names:
        instance_name = f"north-l4-p{i}"
        boot_disk_device_name = f"north-l4-p{i}"

        # Create the VM
        subprocess.run([
            "gcloud", "compute", "instances", "create", instance_name,
            "--project", project,
            "--zone", zone,
            "--machine-type", machine_type,
            "--network-interface", "network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
            "--maintenance-policy=TERMINATE",
            "--provisioning-model=STANDARD",
            "--service-account", service_account,
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            "--accelerator", accelerator,
            "--boot-disk-auto-delete",
            "--boot-disk-size", boot_disk_size,
            "--boot-disk-type", boot_disk_type,
            "--boot-disk-device-name", boot_disk_device_name,
            "--image", image,
            "--image-project", image_project,
            "--no-shielded-secure-boot",
            "--shielded-vtpm",
            "--shielded-integrity-monitoring",
            "--labels=goog-ec-src=vm_add-gcloud",
            "--reservation-affinity=any"
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create VM instances.')
    parser.add_argument('--vm_names', nargs='+', type=int, required=True, help='List of VM numbers to create.')
    parser.add_argument('--zone', type=str, default='us-central1-a', help='Zone to create the VMs in.')
    
    args = parser.parse_args()
    create_vms(args.vm_names, args.zone)
