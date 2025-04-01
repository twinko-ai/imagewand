from setuptools import find_packages, setup

setup(
    name="imagewand",
    # ... other setup parameters ...
    install_requires=["opencv-python", "numpy", "click", "tqdm"],
    entry_points={
        "console_scripts": [
            "imagewand-merge=imagewand.scripts.merge_command:merge",
        ],
    },
)
