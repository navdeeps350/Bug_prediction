import os
import pandas as pd

if __name__ == '__main__':
    
    buggy_class_list = []

    for (dirpath, dirnames, filenames) in os.walk("./resources"):
        for filename in filenames:
            if filename.endswith('.src'):
                with open(os.path.join(dirpath, filename), 'r') as file:
                    buggy_class = file.read()
                    b_list = []
                    for buggy in buggy_class.split('\n'):
                        if buggy != '':
                            parts = buggy.rsplit('.', 1)[-1]
                            b_list.append(parts)
                    buggy_class_list.extend(b_list)

    feature_frame = pd.read_csv('results/feature_vectors.csv')  

    feature_frame['buggy'] = feature_frame['class_name'].apply(lambda buggy_class: 1 if buggy_class in buggy_class_list else 0)

    feature_frame.to_csv('results/label_feature_vectors.csv', index=False)
    print('Labelled feature vectors saved to new_feature_vector.csv in the results folder.')
    print(f'Total number of buggy classes: are {feature_frame['buggy'].sum()} and non-buggy classes are {len(feature_frame) - feature_frame['buggy'].sum()} out of {len(feature_frame)} classes.')