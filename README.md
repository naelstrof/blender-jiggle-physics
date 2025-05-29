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

### Intuitive UI

Jiggles will generally just work, but certain settings (like connected bones) will prevent bones from stretching.

Warnings with descriptive tooltips will appear in these special cases.

![image](https://github.com/user-attachments/assets/2fba2440-f106-4476-8301-f41440b4836a)

### Debug Mode

An easily accessible debug mode allows you to see the system's detected rest pose (in green), and the simulation (in red).

![image](https://github.com/user-attachments/assets/14ce81d1-d3be-49e1-a2e8-37e0c278ab85)

As a bonus, cProfile is enabled with Debug Mode so you can analyze the cost of Jiggle Physics calculations.

## Special Thanks

- shteeve3d for creating Wiggle 2
- Raliv for the awesome verlet solver
- Ewwgine for helping me with Blender and the Giraffe demonstration model.

## License

See [LICENSE](LICENSE)
