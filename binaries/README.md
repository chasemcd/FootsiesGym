# Linux


# Mac

This has only been tested using an M3 chip. Unfortunately, gRPC (specifically Grpc.Core) is not compatible with Silicon Macs, so we have to do some workarounds in order to use the exact workflow from Linux. Download and unzip the Mac build that you're interested in using and run the game servers with:

 ```
 # Windowed Build
 arch --x86_64 footsies_mac_windowed_5709b6d.app/Contents/MacOS/FOOTSIES --port <YOUR_DESIRED_PORT>

 # Headless Server
 arch -x86_64 footsies_mac_headless_5709b6d/FOOTSIES --port <YOUR_DESIRED_PORT>
 ```

 The ports will default to 50051.

 If you run into an error on Mac that says "This will damage your computer," you may need to run (specifically for the headless build):

```
codesign --force --deep --sign - /footsies_mac_headless_5709b6d/FOOTSIES
```
