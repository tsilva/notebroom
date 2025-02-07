import sys
from .core import process_nb, get_output_filename

def main():
    if len(sys.argv) != 2:
        print("Usage: notebroom <notebook.ipynb>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = get_output_filename(input_file)
    process_nb(input_file, output_file)

if __name__ == "__main__":
    main()
