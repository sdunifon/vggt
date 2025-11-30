#!/usr/bin/env python3
"""
test_api.py - Test script for VGGT RunPod API
Tests the 3D reconstruction endpoint with sample images using gradio_client.
"""

from gradio_client import Client, handle_file
import requests
import os

# Hardcoded endpoint and API key
"

# Test images - using kitchen examples
IMAGE_PATHS = [
    "examples/kitchen/images/00.png",
    "examples/kitchen/images/01.png",
]

# Output file
OUTPUT_FILE = "test_output.ply"


def test_ping():
    """Test the /ping endpoint."""
    print("Testing /ping endpoint...")
    response = requests.get(
        f"{ENDPOINT}/ping",
        headers={"Authorization": API_KEY}
    )
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text}")
    return response.status_code == 200


def test_vggt_api():
    """Test the VGGT 3D reconstruction API using gradio_client."""
    print("\nTesting VGGT API with gradio_client...")

    # Create client with auth header
    print(f"  Connecting to {ENDPOINT}...")
    client = Client(
        ENDPOINT,
        headers={"Authorization": API_KEY}
    )

    print(f"  Client connected!")

    # Prepare file handles for upload
    print(f"  Preparing {len(IMAGE_PATHS)} images...")
    files = [handle_file(path) for path in IMAGE_PATHS]

    # Call the API
    print("  Calling vggt/create/images API (this may take a while)...")
    result = client.predict(
        images=files,
        output_format="ply",
        conf_thres=50,
        api_name="/vggt/create/images"
    )

    print(f"  Result: {result}")

    # Result should be (output_file_path, analytics_string)
    if result and len(result) >= 1:
        output_path = result[0]
        analytics = result[1] if len(result) > 1 else ""

        print(f"\n  Output path: {output_path}")
        print(f"  Analytics: {analytics}")

        # Copy or download the result
        if output_path and os.path.exists(output_path):
            import shutil
            shutil.copy(output_path, OUTPUT_FILE)
            print(f"\n  Copied to {OUTPUT_FILE}")
            print(f"  File size: {os.path.getsize(OUTPUT_FILE)} bytes")
            return True
        elif isinstance(output_path, str) and output_path.startswith("http"):
            # Download from URL
            response = requests.get(
                output_path,
                headers={"Authorization": API_KEY}
            )
            if response.status_code == 200:
                with open(OUTPUT_FILE, "wb") as f:
                    f.write(response.content)
                print(f"\n  Downloaded to {OUTPUT_FILE}")
                print(f"  File size: {len(response.content)} bytes")
                return True

    return False


def main():
    print("=" * 50)
    print("VGGT RunPod API Test")
    print("=" * 50)
    print(f"Endpoint: {ENDPOINT}")
    print(f"Images: {IMAGE_PATHS}")
    print()

    # Test ping
    if not test_ping():
        print("\nPing failed! Server may not be ready.")
        return

    # Test VGGT API
    try:
        success = test_vggt_api()
    except Exception as e:
        print(f"\nError: {e}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: 3D reconstruction completed!")
        print(f"Output saved to: {OUTPUT_FILE}")
    else:
        print("FAILED: Could not complete 3D reconstruction")
    print("=" * 50)


if __name__ == "__main__":
    main()
