CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case1/input.png --output_path test_results/case1
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case1/panorama.png --classes outdoor --output_path test_results/case1

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case3/input.png --output_path test_results/case3
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case3/panorama.png --classes outdoor --output_path test_results/case3

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "There is a rocky island on the vast sea surface, with a triangular rock burning red flames in the center of the island. The sea is open and rough, with a green surface. Surrounded by towering peaks in the distance." --output_path test_results/case4
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case4/panorama.png --classes outdoor --output_path test_results/case4

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case5/input.png --output_path test_results/case5
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case5/panorama.png --classes outdoor --output_path test_results/case5

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case6/input.png --output_path test_results/case6
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case6/panorama.png --labels_fg1 tent --classes outdoor --output_path test_results/case6

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "" --image_path examples/case8/input.png --output_path test_results/case8
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case8/panorama.png --classes outdoor --output_path test_results/case8

CUDA_VISIBLE_DEVICES=0 python3 demo_panogen.py --prompt "A breathtaking volcanic eruption scene. In the center of the screen, one or more volcanoes are erupting violently, with hot orange red lava gushing out from the crater, illuminating the surrounding night sky and landscape. Thick smoke and volcanic ash rose into the sky, forming a huge mushroom cloud like structure. Some of the smoke and dust were reflected in a dark red color by the high temperature of the lava, creating a doomsday atmosphere. In the foreground, a winding lava flow flows through the dark and rough rocks like a fire snake, emitting a dazzling light as if burning the earth. The steep and rugged mountains in the background further emphasize the ferocity and irresistible power of nature. The entire picture has a strong contrast of light and shadow, with red, black, and gray as the main colors, highlighting the visual impact and dramatic tension of volcanic eruptions, making people feel the grandeur and terror of nature." --output_path test_results/case9
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case9/panorama.png --classes outdoor --output_path test_results/case9
