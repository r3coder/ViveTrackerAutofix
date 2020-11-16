# ViveTrackerAutofix
This code automatically fixes pelvis tracker's position using simple Neural network.

**Still working on integration with OpenVR Input emulator to apply data to the tracker, and optimizing the network.**

Uploaded to Github to share idea if it is possible.

## Participators
@r3coder, @jshparksh for writing code and report

### Disclaimer

Large amount of this repository is based on VRPlayspaceMover by naelstrof (https://github.com/naelstrof/VRPlayspaceMover/).

- Loading OpenVR and connect into it
- Updating tracking devices' position
- Handling Exit Signal
- and etc...

### Compile method

go to https://github.com/naelstrof/VRPlayspaceMover/, and follow compile instructions on there. Then, replace `PlayspaceMover.cpp`'s content to `Network.cpp`'s content. Then, it will compile. Please...

### How to execute

Currently, this repository is 90% for personal use, so it may uncomfortable to other people to use.

- Turn on SteamVR without turning on controllers or trackers.
- Turn on controllers
- Turn on foot trackers
- Lastly, turn on pelvis tracker (or hip tracker)
- Then, compile this code.

Later on, if I have enough time, I'll working on to find other tracking devices automatically.
