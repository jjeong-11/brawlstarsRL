from pathlib import Path
import cv2

from getImage import get_image
from getAnchor import find_player_position # type: ignore
from getCube import find_cube_info # type: ignore
from getHealth import find_health_info # type: ignore
from getAmmo import find_ammo_info # type: ignore
from getEnemies import find_enemies_with_stats, find_boxes # type: ignore
from getGas import is_player_in_gas # type: ignore
from getPickups import find_ground_cubes # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "debugOutput"


def get_next_debug_filename():
    """Returns debugOutput/debug_output_1.png, debug_output_2.png, ..."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        filename = OUTPUT_DIR / f"debug_output_{i}.png"

        if not filename.exists():
            return str(filename)

        i += 1
def main():
    # Ask the user for an image
    image, image_path = get_image()

    #plug in the image to find player position
    player_x, player_y, anchor_radius = find_player_position(image)

    #plug in the image and player position to find cube info
    anchor = (player_x, player_y, anchor_radius)
    cube_info = find_cube_info(image, anchor)

    health_info = find_health_info(image, (player_x, player_y, anchor_radius))

    ammo_info = find_ammo_info(image, (player_x, player_y, anchor_radius))

    # find_enemies_with_stats already drops any candidate where neither
    # health nor cube count could be read, so everything returned here
    # is treated as a validated detection. Passing the player's own anchor
    # lets it drop candidates that are actually the local player's own
    # (sometimes red-segmented) HP bar rather than a real enemy.
    enemies = find_enemies_with_stats(image, player_pos=anchor)

    in_gas = is_player_in_gas(image, anchor)

    boxes = find_boxes(image, player_pos=anchor)

    # Exclude the player and every detected enemy/box position so their
    # own green-and-yellow sprites can't be mistaken for loose cubes.
    exclude = [anchor]
    exclude += [(e["center"][0], e["center"][1], e["radius"]) for e in enemies]
    exclude += [(b["center"][0], b["center"][1], b["radius"]) for b in boxes]
    ground_cubes = find_ground_cubes(image, exclude_positions=exclude)

    #debugging: draw the player position and cube bounding box on the image and save it
    debug = image.copy()
    # Draw player position
    cv2.circle(debug, (player_x, player_y), 6, (0, 0, 255), -1)

    # Draw detected player radius
    cv2.circle(
        debug,
        (player_x, player_y),
        int(anchor_radius),
        (255, 0, 0),
        2,
    )

    cv2.putText(
        debug,
        "Player",
        (player_x + 10, player_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # Draw cube bounding box
    if cube_info["bounding_box"] is not None:

        x, y, w, h = cube_info["bounding_box"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2,
        )

        label = f"Cubes: {cube_info['cube_count']}"

        cv2.putText(
            debug,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    # Draw health bounding box (Bright Green)
    if health_info["bounding_box"] is not None:
        hx, hy, hw, hh = health_info["bounding_box"]
        cv2.rectangle(
            debug,
            (hx, hy),
            (hx + hw, hy + hh),
            (0, 255, 0),
            2,
        )
        hp_label = f"HP: {health_info['current_health']}"
        # Placed slightly below the box if it fights with username space, 
        # or above (hy - 8) like the cube counter
        cv2.putText(
            debug,
            hp_label,
            (hx, hy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    if ammo_info["bounding_box"] is not None:
        ax, ay, aw, ah = ammo_info["bounding_box"]
        cv2.rectangle(debug, (ax, ay), (ax + aw, ay + ah), (0, 69, 255), 2)

        ammo_label = f"Ammo: {ammo_info['ammo_count']}"
        cv2.putText(
            debug,
            ammo_label,
            (ax, ay + ah + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 69, 255),
            2,
        )

    # Draw every validated enemy target, plus their read HP / cube stats
    for index, enemy in enumerate(enemies):
        ex, ey = enemy["center"]
        er = enemy["radius"]
        ebx, eby, ebw, ebh = enemy["bounding_box"]
        health_info_e = enemy["health"]
        cube_info_e = enemy["cubes"]

        # Draw the target center marker (Crimson Red)
        cv2.circle(debug, (ex, ey), 5, (0, 0, 255), -1)

        # Draw the tracking bounding box perimeter
        cv2.rectangle(debug, (ebx, eby), (ebx + ebw, eby + ebh), (0, 0, 255), 2)

        # Label the targets sequentially, including whatever stats were read
        hp_text = health_info_e["current_health"]
        cube_text = cube_info_e["cube_count"]
        cv2.putText(
            debug,
            f"Enemy #{index+1} HP:{hp_text} Cubes:{cube_text}",
            (ebx, eby - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # Draw the enemy's health bar box (magenta, to distinguish from the player's green one)
        if health_info_e["bounding_box"] is not None:
            hx, hy, hw, hh = health_info_e["bounding_box"]
            cv2.rectangle(debug, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)

        # Draw the enemy's cube badge box (cyan, to distinguish from the player's yellow one)
        if cube_info_e["bounding_box"] is not None:
            cx, cy, cw, ch = cube_info_e["bounding_box"]
            cv2.rectangle(debug, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)

    # Draw destructible boxes (orange) with their HP
    for index, box in enumerate(boxes):
        bx, by, bw, bh = box["bounding_box"]
        cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
        cv2.putText(
            debug,
            f"Box HP:{box['health']['current_health']}",
            (bx, by - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
        if box["health"]["bounding_box"] is not None:
            hx, hy, hw, hh = box["health"]["bounding_box"]
            cv2.rectangle(debug, (hx, hy), (hx + hw, hy + hh), (0, 165, 255), 1)

    # Draw ground cubes (bright green)
    for cube in ground_cubes:
        cx_, cy_, cw_, ch_ = cube["bounding_box"]
        cv2.rectangle(debug, (cx_, cy_), (cx_ + cw_, cy_ + ch_), (0, 255, 0), 2)
        label = "Cube pile" if cube["clustered"] else "Cube"
        cv2.putText(
            debug,
            label,
            (cx_, cy_ - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Flag whether the player is standing in the poison gas
    if in_gas:
        cv2.putText(
            debug,
            "IN GAS!",
            (player_x - 40, player_y + int(anchor_radius) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 130, 0),
            3,
        )

    filename = get_next_debug_filename()
    cv2.imwrite(filename, debug)
    # Print the results
    print(f"\nImage: {image_path}")
    print(f"Player position: ({player_x}, {player_y}, {anchor_radius})")
    print(f"Cube info: {cube_info}")
    print(f"Health info: {health_info}")
    print(f"Ammo info: {ammo_info}")
    print(f"Player in gas: {in_gas}")
    print(f"Boxes found: {len(boxes)}")
    for index, box in enumerate(boxes):
        print(f"  Box #{index+1}: center={box['center']} health={box['health']}")
    print(f"Ground cubes found: {len(ground_cubes)}")
    for index, cube in enumerate(ground_cubes):
        print(f"  Cube #{index+1}: center={cube['center']} clustered={cube['clustered']}")
    print(f"Validated Enemy Count: {len(enemies)}")
    for index, enemy in enumerate(enemies):
        print(
            f"  Enemy #{index+1}: center={enemy['center']} "
            f"health={enemy['health']} cubes={enemy['cubes']}"
        )
    print(f"Debug image saved as {filename}")
    print(f"\n")

if __name__ == "__main__":
    main()