import pickle
if __name__ == "__main__":
    with open('y_error_graph_data_final.pkl', 'rb') as f:
        test = pickle.load(f)
        debug_var = 1