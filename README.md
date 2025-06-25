# Jiggle Physics

Jiggle Physics is a fork of the Wiggle 2 addon, with a variety of improvements.

## Installation

Ensure you're using Blender 4.2 or higher. Then download the latest zip from [Releases](https://github.com/naelstrof/blender-jiggle-physics/releases/latest), install using your user preferences!

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

### Overlays

Easily accessible overlays allow you to see the system's detected rest pose (in green), and the simulation (in red).

![double view with only one showing the overlay](https://github.com/user-attachments/assets/14ce81d1-d3be-49e1-a2e8-37e0c278ab85)

You can use this to visualize collision radius of bones:

![collision radius on a dragon's tail and ears](https://github.com/user-attachments/assets/02a8ee21-f650-4809-8be3-12f81cbdcea7)

## Usage

- Find Jiggle Physics under the Animation tab by pressing the N key in the 3D viewport.

![Animation tab](https://github.com/user-attachments/assets/839fcf23-f756-411f-aa3c-77a669a74d05)

- Enable the scene's jiggle by clicking the closed eyeball.

- Enable a pose bone's jiggle by selecting a bone in pose mode and checking the "Jiggle Bone" checkbox.

![parameter list](https://github.com/user-attachments/assets/ff955f3e-c747-48b0-8ff2-87e5a96bb280)

- Add a position and rotation keyframe with the `i` key on the bones to set the "rest" pose of the jiggle. 

## Troubleshooting

Keep an eye out for warning buttons in the panel, they will detect most issues and describe how to solve them automatically!

![warning popup](https://github.com/user-attachments/assets/2fba2440-f106-4476-8301-f41440b4836a)

### It looks perfect during playback, but stops working during render!

This is by design, you must "bake" the jiggles before rendering them.

This is because Blender can render frames out of order by design. To prevent unintended jiggles, jiggle physics always disables itself during render. This also makes the addon "safe" to add to render farms.

### My bones are drooping away, and something is clearly wrong.

![droopy solve](https://github.com/user-attachments/assets/38d499f0-4ff3-452f-a088-ee3e2453d4e0)

Without a position and rotation keyframe, Jiggle Physics doesn't know what the "rest" pose should be! Double check your dope sheet to ensure that the jiggle bones have position and rotation keyframes. If they don't, reset their position and rotation with *alt+g* and *alt+r*, then add a keyframe with *i*.

### My settings don't seem to stick after changing them!

![keyframed parameters](https://github.com/user-attachments/assets/fcf91027-708a-4557-b556-9520892b5594)

The jiggle parameters can be keyframed into animations and actions. There's a handy button to delete all jiggle parameter keys for all selected bones called *"Clear Parameter Keyframes"* which will attempt to delete them for you. If your keyframes are in an action, make sure that you're tweaking the action first!

### The pose doesn't quite match the simulation.

Jiggle Physics needs to predict the relationship between bones, and it uses standard parenting to try to understand.

Some more complex rigs might use constraints, bendy bones, or special parenting rules (like ignoring parent positioning) and they aren't supported well.

Try only jiggling deform bones!

### I'm not getting any stretchy physics on my bones!

"Connected" bones cannot be translated. This prevents jiggle physics from stretching them. If your armature is linked in as an asset then you might need to edit the source file.

### How do I set a bone to have a 90 degree rotation limit?

That's the neat part, you don't! If you are having trouble with your bones moving too much with your desired *elasticity*, you can increase *elasticity* to limit the motion and use the *elasticity soften* parameter to soften motion near the target pose

### How do I collide with a flat surface?

Unfortunately meshes only work okay with bones that have a large radius. Flat planes are infinitely thin. Instead you should try to build your colliders out of scaled empties (preferrably *Empty->Sphere*s). One way to get a flat floor is to make a planetary-sized collider below the feet.

## Special Thanks

- shteeve3d for creating Wiggle 2
- Raliv for the awesome verlet solver
- Ewwgine for helping me with Blender and the Giraffe demonstration model.

## License

See [LICENSE](LICENSE)
