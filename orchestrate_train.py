import os

COPY_TO_CLIPBOARD = True
ONLY_ONE_EPOCH = False
train_script = 'quick_train.py'


config_files = []

command = ""

for config_file in config_files:
    print(f'Running {train_script} with config: {config_file}')
    command += f'python {train_script} {config_file}'
    command += ' --quick' if ONLY_ONE_EPOCH else ''
    command += ';'


print("Command:", command.strip())

if COPY_TO_CLIPBOARD:
    # Copy the command to clipboard
    os.system(f'echo "{command.strip()}" | clip.exe')
    print("Command copied to clipboard.")
else:
    # Execute the command
    os.system(command.strip())
    