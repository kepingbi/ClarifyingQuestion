'''Utility functions shared by the whole project
'''
import gzip
import torch
from others.logging import logger
from collections import defaultdict

def pad_sequences(sequences):
    lengths = [len(seq) for seq in sequences]

    padded_sequences = torch.zeros(len(sequences), max(lengths)).long()  # padding value = 0

    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_sequences[i, :end] = seq[:end]

    return padded_sequences, lengths

def pad(data, pad_id, width=-1):
    ''' pad dim 1 of a 2-d tensor
    '''
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [d[:width] + [pad_id] * (width - len(d)) for d in data]
    #if width < max(len(d)) of data
    return rtn_data

def left_pad(data, pad_id, width=-1):
    ''' pad dim1 with pad_id to the left of a 2-d tensor
    '''
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [[pad_id] * (width - len(d)) + d[:width] for d in data]
    #if width < max(len(d)) of data
    return rtn_data

def pad_3d(data, pad_id, dim=1, width=-1):
    ''' pad a 3-d tensor to the right
    '''
    #dim = 1 or 2
    dim2 = 0
    for entry in data:
        if len(entry) > 0:
            dim2 = max(dim2, max(len(d) for d in entry))

    if dim < 1 or dim > 2:
        return data
    if width == -1:
        if dim == 1:
            #dim 0,2 is same across the batch
            width = max(len(d) for d in data)
        elif dim == 2:
            #dim 0,1 is same across the batch
            for entry in data:
                width = max(width, max(len(d) for d in entry))
        #print(width)
    if dim == 1:
        rtn_data = [d[:width] + [[pad_id] * dim2] * (width - len(d)) for d in data]
    elif dim == 2:
        rtn_data = []
        for entry in data:
            rtn_data.append([d[:width] + [pad_id] * (width - len(d)) for d in entry])
    return rtn_data

def left_pad_3d(data, pad_id, dim=1, width=-1):
    ''' pad a 3-d tensor to the left
    '''
    #dim = 1 or 2
    dim2 = 0
    for entry in data:
        if len(entry) > 0:
            dim2 = max(dim2, max(len(d) for d in entry))

    if dim < 1 or dim > 2:
        return data
    if width == -1:
        if dim == 1:
            #dim 0,2 is same across the batch
            width = max(len(d) for d in data)
        elif dim == 2:
            #dim 0,1 is same across the batch
            # for entry in data:
            # width = max(width, max(len(d) for d in entry))
            width = dim2
            # print(width)
    if dim == 1:
        # rtn_data = [[[pad_id] * len(data[0][0])] * (width - len(d)) + d[:width] for d in data]
        rtn_data = [[[pad_id] * dim2] * (width - len(d)) + d[:width] for d in data]
    elif dim == 2:
        rtn_data = []
        for entry in data:
            rtn_data.append([[pad_id] * (width - len(d)) + d[:width] for d in entry])
    return rtn_data


def pad_4d_dim1(data, pad_id, width=-1):
    ''' pad dim1 of a 4-d tensor to the right
    '''
    dim3 = 0
    for d_0 in data:
        if len(d_0) > 0:
            for d_1 in d_0:
                if len(d_1) > 0:
                    dim3 = max(dim3, max(len(d_2) for d_2 in d_1))
    if width == -1:
        #max width of dim1
        width = max(width, max(len(d) for d in data))
    #print(width)
    # rtn_data = [d[:width] + [[[pad_id] * len(data[0][0][0])]] * (width - len(d)) for d in data]
    # it requires the last dimension are the same, such as the embedding size
    # otherwise, we need to change to the following
    rtn_data = [d[:width] + [[[pad_id] * dim3]] * (width - len(d)) for d in data]
    return rtn_data

def left_pad_4d_dim1(data, pad_id, width=-1):
    ''' pad dim1 of a 4-d tensor to the left
    '''
    dim3 = 0
    for d_0 in data:
        if len(d_0) > 0:
            for d_1 in d_0:
                if len(d_1) > 0:
                    dim3 = max(dim3, max(len(d_2) for d_2 in d_1))
    if width == -1:
        #max width of dim1
        width = max(width, max(len(d) for d in data))
    #print(width)
    rtn_data = [[[[pad_id] * dim3]] * (width - len(d)) + d[:width] for d in data]
    return rtn_data

def pad_4d_dim2(data, pad_id, width=-1):
    #only handle padding to dim = 2
    dim3 = 0
    for d_0 in data:
        if len(d_0) > 0:
            for d_1 in d_0:
                if len(d_1) > 0:
                    dim3 = max(dim3, max(len(d_2) for d_2 in d_1))
    if width == -1:
        #max width of dim2
        for entry in data:
            width = max(width, max(len(d) for d in entry))
    #print(width)
    rtn_data = []
    for entry_dim1 in data:
        rtn_data.append([d[:width] + [[pad_id] * dim3] * (width - len(d)) \
            for d in entry_dim1])
    return rtn_data

def pad_4d_dim3(data, pad_id, width=-1):
    #only handle padding to dim = 2
    dim3 = 0     
    for d_0 in data:
        if len(d_0) > 0:
            for d_1 in d_0:
                if len(d_1) > 0:
                    dim3 = max(dim3, max(len(d_2) for d_2 in d_1))
    if width == -1:
        width = dim3
    #print(width)
    rtn_data = []
    for entry_dim1 in data:
        rtn_data.append(pad_3d(entry_dim1, pad_id, dim=2, width=width))
        
    return rtn_data

def left_pad_4d_dim2(data, pad_id, width=-1):
    #only handle padding to dim = 2 to the left
    dim3 = 0
    for d_0 in data:
        if len(d_0) > 0:
            for d_1 in d_0:
                if len(d_1) > 0:
                    dim3 = max(dim3, max(len(d_2) for d_2 in d_1))
    if width == -1:
        #max width of dim2
        for entry in data:
            width = max(width, max(len(d) for d in entry))
    #print(width)
    # in case all the elements are empty, whether it is needed?
    # if dim3 == 0:
    #     dim3 = 1
    rtn_data = []
    for entry_dim1 in data:
        rtn_data.append([[[pad_id] * dim3] * (width - len(d)) + d[:width] \
            for d in entry_dim1])
    return rtn_data


def main():
    data = [[[[2, 2, 2], [2, 2, 2]], [[2, 2, 2]]], [[[2, 2, 2]]]]
    rtn = pad_4d_dim1(data, -1)
    rtn = pad_4d_dim2(rtn, -1)
    print(rtn)

if __name__ == "__main__":
    main()
