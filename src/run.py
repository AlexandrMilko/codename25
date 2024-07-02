from __init__ import app

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", debug=False)
    except OSError as e:
        print(f"WARNING: {e}")
        print(f"WARNING: trying to start on different port, 5001")
        app.run(host="0.0.0.0", port=5001, debug=False)