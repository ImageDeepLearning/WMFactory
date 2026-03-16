import trimesh


def process_file(input_path, output_path):
    r"""Convert a PLY file to Draco format.
    Args:
        input_path (str): Path to the input PLY file.
        output_path (str): Path to save the output Draco file.
    """
    mesh = trimesh.load(input_path)
    try:
        # Attempt Draco-compressed PLY export
        export_data = trimesh.exchange.ply.export_draco(mesh)
        with open(output_path, 'wb') as f:
            f.write(export_data)
    except Exception as e:
        print(f"Draco export failed: {str(e)}. May need confirm installation")
