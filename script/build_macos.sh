#!/usr/bin/env zsh

if [[ ! -f gui.py ]]; then
	print -u2 "Error: This script must be run from the project root"
	exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
	print -u2 "Error: Virtual environment must exist at '.venv'"
	exit 1
fi

if ! .venv/bin/python -m pip --version > /dev/null; then
	print "Installing pip in virtual environment..."
	if ! .venv/bin/python -m ensurepip --upgrade; then
		print -u2 "Error: failed to install pip"
		exit 1
	fi
fi

print "Installing dependencies..."
.venv/bin/python -m pip install -r ./requirements.txt
if (( $? != 0 )); then
	print -u2 "Error: failed to install dependencies"
	exit 1
fi

print "Installing pyinstaller..."
.venv/bin/python -m pip install pyinstaller==6.22.3
if (( $? != 0 )); then
	print -u2 "Error: failed to install pyinstaller"
	exit 1
fi

print "Running pyinstaller..."
.venv/bin/python -m PyInstaller --noconfirm --windowed --add-data=data:data gui.py --name ToanMachine
if (( $? != 0 )); then
	print -u2 "Error: failed to run pyinstaller"
	exit 1
fi

print "Adding microphone permission..."
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string Toan Machine needs microphone access to record audio from your guitar gear." "./dist/ToanMachine.app/Contents/Info.plist"
if (( $? != 0 )); then
	print -u2 "Error: failed to add microphone permission"
	exit 1
fi

# Updating Info.plist above invalidates PyInstaller's bundle signature.
print "Re-signing app bundle..."
codesign --force --deep --sign - "./dist/ToanMachine.app"
if (( $? != 0 )); then
	print -u2 "Error: failed to sign app bundle"
	exit 1
fi

print "Packaging DMG..."
hdiutil create -volname "ToanMachine" -srcfolder "./dist/ToanMachine.app" -ov -format UDZO "./dist/ToanMachine.dmg"
if (( $? != 0 )); then
	print -u2 "Error: failed to create disk image"
	exit 1
fi
