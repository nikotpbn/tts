"""
launch_training.py — Launch a spot EC2 instance to train an XTTS model.

Usage:
    python scripts/launch_training.py --character thrall
    python scripts/launch_training.py --character thrall --epochs 50 --batch-size 4

Notes:
    - Run this script LOCALLY.
    - Reads infra/userdata.sh, substitutes variables, and passes it to EC2.
    - Requires .env in project root with EC2 configuration.
    - The instance will automatically:
        1. Pull latest code from GitHub
        2. Download dataset from S3
        3. Run training
        4. Upload model to S3
        5. Send SNS notification
        6. Self-terminate
"""

import os
import sys
import argparse
import logging
import base64
from pathlib import Path
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch EC2 spot instance for XTTS training."
    )
    parser.add_argument("--character", type=str, required=True, help="Character name")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default: 2)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("launch_training")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Userdata
# ---------------------------------------------------------------------------


def load_userdata(
    project_root: Path, character: str, epochs: int, batch_size: int, env: dict
) -> str:
    """Load userdata.sh template and substitute placeholders."""
    template_path = project_root / "infra" / "userdata.sh"

    if not template_path.exists():
        raise FileNotFoundError(f"userdata.sh not found at {template_path}")

    template = template_path.read_text()

    substitutions = {
        "{{CHARACTER}}": character,
        "{{EPOCHS}}": str(epochs),
        "{{BATCH_SIZE}}": str(batch_size),
        "{{S3_BUCKET}}": env["S3_BUCKET"],
        "{{GITHUB_REPO}}": env["GITHUB_REPO"],
        "{{SNS_TOPIC_ARN}}": env["SNS_TOPIC_ARN"],
        "{{CLOUDWATCH_LOG_GROUP}}": env["CLOUDWATCH_LOG_GROUP"],
        "{{AWS_DEFAULT_REGION}}": env["AWS_DEFAULT_REGION"],
    }

    for placeholder, value in substitutions.items():
        template = template.replace(placeholder, value)

    return template


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch(character: str, epochs: int, batch_size: int, logger: logging.Logger):
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")

    required = [
        "S3_BUCKET",
        "AMI_ID",
        "INSTANCE_TYPE",
        "SUBNET_ID",
        "SECURITY_GROUP_ID",
        "INSTANCE_PROFILE_ARN",
        "SPOT_MAX_PRICE",
        "SNS_TOPIC_ARN",
        "CLOUDWATCH_LOG_GROUP",
        "GITHUB_REPO",
        "AWS_DEFAULT_REGION",
    ]

    env = {k: os.getenv(k) for k in required}
    missing = [k for k, v in env.items() if not v]

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Check your .env file.")
        sys.exit(1)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    logger.info(f"Character:      {character}")
    logger.info(f"Run ID:         {run_id}")
    logger.info(f"AMI:            {env['AMI_ID']}")
    logger.info(f"Instance type:  {env['INSTANCE_TYPE']}")
    logger.info(f"Max spot price: ${env['SPOT_MAX_PRICE']}/hr")
    logger.info(f"Epochs:         {epochs}")
    logger.info(f"Batch size:     {batch_size}")

    # Load and substitute userdata
    try:
        userdata = load_userdata(project_root, character, epochs, batch_size, env)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    userdata_b64 = base64.b64encode(userdata.encode("utf-8")).decode("utf-8")

    try:
        ec2 = boto3.client("ec2", region_name=env["AWS_DEFAULT_REGION"])

        response = ec2.run_instances(
            ImageId=env["AMI_ID"],
            InstanceType=env["INSTANCE_TYPE"],
            MinCount=1,
            MaxCount=1,
            UserData=userdata_b64,
            SubnetId=env["SUBNET_ID"],
            SecurityGroupIds=[env["SECURITY_GROUP_ID"]],
            IamInstanceProfile={"Arn": env["INSTANCE_PROFILE_ARN"]},
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "MaxPrice": env["SPOT_MAX_PRICE"],
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"tts-training-{character}-{run_id}"},
                        {"Key": "Character", "Value": character},
                        {"Key": "RunId", "Value": run_id},
                        {"Key": "Project", "Value": "tts-pipeline"},
                    ],
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 100,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
        )

        instance_id = response["Instances"][0]["InstanceId"]

        logger.info(f"Instance launched: {instance_id}")
        logger.info(
            f"Monitor logs:      aws logs tail {env['CLOUDWATCH_LOG_GROUP']} --follow"
        )
        logger.info(
            f"S3 model path:     s3://{env['S3_BUCKET']}/characters/{character}/models/"
        )
        logger.info("You will receive an email when training completes or fails.")

    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to launch instance: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    launch(
        character=args.character,
        epochs=args.epochs,
        batch_size=args.batch_size,
        logger=logger,
    )
