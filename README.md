# Jiggle Physics

Jiggle Physics is a fork of the Wiggle 2 addon, with a variety of improvements.

## Installation

Download the latest zip from [Releases](https://github.com/naelstrof/blender-jiggle-physics/releases/latest), install using your user preferences!

## Features

### Design at runtime

- Reasonably performant within the viewport, allowing you to tweak jiggles while your animation plays!
  
[Screencast_20250528_190839.webm](https://github.com/user-attachments/assets/156016e3-7f77-48b9-98e7-130dfcefb854)


### Robust verlet physics solve

- Simple 0-1 parameters. Jiggle anything from soupy ropes to immovable rods.
- Verlet integration makes the system incredibly resilient to "exploding". 
- Supports bone squash and stretch.
- Relativistic, reacts appropriately to elevators and vehicles.

Uses the same solver as [Unity Jiggle Physics](https://github.com/naelstrof/UnityJigglePhysics)!

### Collision support

- Supports good collision with scaled empties, and limited support for mesh collision. Unfortunately doesn't support bouncing or friction.

[Screencast_20250531_141235.webm](https://github.com/user-attachments/assets/982a9b62-2b65-44ae-9f07-6b0b5650067d)

### Debug Mode

An easily accessible debug mode allows you to see the system's detected rest pose (in green), and the simulation (in red).

![image](https://github.com/user-attachments/assets/14ce81d1-d3be-49e1-a2e8-37e0c278ab85)

As a bonus, cProfile is enabled with Debug Mode so you can analyze the cost of Jiggle Physics calculations.

## Usage

Find Jiggle Physics under the Animation tab by pressing the N key in the 3D viewport.

![Animation tab](https://github.com/user-attachments/assets/839fcf23-f756-411f-aa3c-77a669a74d05)

Enable the scene's jiggle by clicking the closed eyeball.

Then, selecting bones in pose mode will allow you to enable Jiggle, and adjust their parameters. Be weary that the settings won't appear unless one of the bones are "active" (Box selection does not set an active pose bone).

Adjusting any parameter will batch-change all the selected bones, if keyframe recording is enabled this will also keyframe every bone's parameter.

![parameter list](https://github.com/user-attachments/assets/ff955f3e-c747-48b0-8ff2-87e5a96bb280)

Add a keyframe for the jiggled bones to set their "rest" pose, you can keyframe animation too to make tails wag, and ears perk.

Finally, ensure to read warnings on the panel and try to fix them. For example if you forgot to keyframe a rest pose, it will display a warning button with a tooltip describing the issue and how to fix it:

![warning popup](https://github.com/user-attachments/assets/2fba2440-f106-4476-8301-f41440b4836a)

## Special Thanks

- shteeve3d for creating Wiggle 2
- Raliv for the awesome verlet solver
- Ewwgine for helping me with Blender and the Giraffe demonstration model.

## License

See [LICENSE](LICENSE)
