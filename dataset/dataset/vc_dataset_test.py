from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class VCDataset(Dataset):
    def __init__(self, datasetA, n_frames=64, max_mask_len=25, valid=False):
        self.datasetA = datasetA
        #self.datasetB = datasetB
        self.n_frames = n_frames
        self.valid = valid
        self.max_mask_len = max_mask_len

    def __getitem__(self, index):
        dataset_A = self.datasetA
        #dataset_B = self.datasetB
        n_frames = self.n_frames
        
        # if self.valid:
        #     if dataset_B is None:  # only return datasetA utterance
        #         return dataset_A[index]
        #     else:
        #         return dataset_A[index], dataset_B[index]

        self.length = len(dataset_A)
        num_samples = len(dataset_A)
        #print(self.length)

        test_data_A_idx = np.arange(len(dataset_A))
        #test_data_B_idx = np.arange(len(dataset_B))
        np.sort(test_data_A_idx)
        #np.sort(test_data_B_idx)
        test_data_A_idx_subset = test_data_A_idx[:num_samples]
        #test_data_B_idx_subset = test_data_B_idx[:num_samples]

        test_data_A = list()
        test_mask_A = list()
        #test_data_B = list()
        #test_mask_B = list()
        #print("loooooooopppp111")
        '''
        for idx_A, idx_B in zip(test_data_A_idx_subset, test_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            #print(frames_A_total)
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            mask_size_A = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_A
            mask_start_A = np.random.randint(0, n_frames - mask_size_A)
            mask_A = np.ones_like(data_A[:, start_A:end_A])
            mask_A[:, mask_start_A:mask_start_A + mask_size_A] = 0.
            test_data_A.append(data_A[:, start_A:end_A])
            test_mask_A.append(mask_A)

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            mask_size_B = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_B
            mask_start_B = np.random.randint(0, n_frames - mask_size_B)
            mask_B = np.ones_like(data_A[:, start_A:end_A])
            mask_B[:, mask_start_B:mask_start_B + mask_size_B] = 0.
            test_data_B.append(data_B[:, start_B:end_B])
            test_mask_B.append(mask_B)
            '''

        for idx_A in test_data_A_idx_subset:
            print("loooooooopppp")
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            #print(frames_A_total)
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            mask_size_A = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_A
            mask_start_A = np.random.randint(0, n_frames - mask_size_A)
            mask_A = np.ones_like(data_A[:, start_A:end_A])
            mask_A[:, mask_start_A:mask_start_A + mask_size_A] = 0.
            test_data_A.append(data_A[:, start_A:end_A])
            test_mask_A.append(mask_A)



        test_data_A = np.array(test_data_A)
        #test_data_B = np.array(test_data_B)
        test_mask_A = np.array(test_mask_A)
        #test_mask_B = np.array(test_mask_B)

        return test_data_A[index], test_mask_A[index]#,  test_data_B[index], test_mask_B[index]

    def __len__(self):
        if self.datasetA is None:
            return len(self.datasetA)
        else:
            return len(self.datasetA)


if __name__ == '__main__':
    # Trivial test for dataset class
    testA = np.random.randn(162, 24, 554)
    testB = np.random.randn(158, 24, 554)
    dataset = VCDataset(testA, testB)
    testLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=False)
    for i, (A, mask_A, B, mask_B) in enumerate(testLoader):
        print(A.shape, mask_B.shape, B.shape, mask_B.shape)
        assert A.shape == mask_B.shape == B.shape == mask_B.shape
        break
