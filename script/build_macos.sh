#!/usr/bin/env zsh

if [[ ! -f gui.py ]]; then
  print -u2 "Error: This script must be run from the project root"
  exit 1
fi

if [[ ! -d .venv ]]; then
  print -u2 "Error: Virtual environment must exist at '.venv'"
  exit 1
fi

. .venv/bin/activate
print "Virtualenv loaded"

pip > /dev/null
if (( $? != 0 )); then
  print -u2 "Error: pip must be installed"
  exit 1
fi

print "Installing dependencies..."
pip install -r ./requirements.txt > /dev/null
if (( $? != 0 )); then
  print -u2 "Error: failed to install dependencies"
  exit 1
fi

print "Installing pyinstaller..."
pip install pyinstaller==6.22.3 > /dev/null
if (( $? != 0 )); then
  print -u2 "Error: failed to install pyinstaller"
  exit 1
fi

print "Running pyinstaller..."
pyinstaller --windowed --add-data=data:data gui.py --name ToanMachine
if (( $? != 0 )); then
  print -u2 "Error: failed to run pyinstaller"
  exit 1
fi

print "Packaging DMG..."
hdiutil create -volname "ToanMachine" -srcfolder "./dist/ToanMachine.app" -ov -format UDZO "./dist/ToanMachine.dmg"
if (( $? != 0 )); then
  print -u2 "Error: failed to create disk image"
  exit 1
fi
