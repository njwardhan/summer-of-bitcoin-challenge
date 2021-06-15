__author__ = 'Nishant Joywardhan'
__email__ = 'njwardhan@gmail.com'


''' STEP-1 : Importing the required libraries '''
import pandas as pd
import numpy as np
from numba import jit


''' STEP-2 : Reading the 'mempool.csv' file as a Pandas dataframe (for easy data-handling) '''
bitcoin_df = pd.read_csv('mempool.csv')


''' STEP-3 : Preparing arrays for the Knapsack appraoch, excluding the txids whose parent-transactions don't appear before them '''
fee_array, weight_array, txids = [], [], []

# Iterating over the rows of the dataframe and dropping the invalid ones
for rows in bitcoin_df.iterrows():
    parents = rows[1][3]

    # For cases having a parent transaction
    if isinstance(parents, str):
        parent_ids = parents.split(';')
        if set(parent_ids).issubset(set(txids)):
        # If the parent transactions are there in the txids list, it is considered, else not.
            txids.append(rows[1][0])
            fee_array.append(rows[1][1])
            weight_array.append(rows[1][2])

    # For cases not having a parent transaction
    else:
        txids.append(rows[1][0])
        fee_array.append(rows[1][1])
        weight_array.append(rows[1][2])

# Converting the 'fee' and 'weight' lists into numpy arrays for better handling (splitting them into sub-arrays)
numpy_fee_split = np.split(np.array(fee_array), 42)
numpy_weight_split = np.split(np.array(weight_array), 42)


'''
Step-4 : "Partial-Knapsack" Approach

After dropping the invalid transactions, we are left with arrays having 4746 elements. With such huge arrays and a given
maximum weight of 4000000 (order of 10^6), applying the direct Knapsack appraoch would not bear any fruitful result.

Thus, the partial approach is based on the assumption that, 

    ==> "Picking the best transactions out of a pool of 4746 options, such that the maximum weight is <= 4000000"

    is same as, (dividing the above inequality by 42 on both sides)

    ==> "Picking the best transactions out of a pool of 113 options, such that the maximum weight is <= 95238" * 42

So, basically, instead of performing Knapsack single time, we would do it multiple times on smaller arrays.
The number 42 is chosen just for the sake of perfect divisibilty.
'''

# Using the Numba compiler for speedy execution
@jit

# `printknapSack` function code credit: https://www.geeksforgeeks.org/printing-items-01-knapsack/; Author: Aryan Garg 
#####
def printknapSack(W, wt, val, n):
    chosen_weights = []
    K = [[0 for w in range(W + 1)] for i in range(n + 1)]
             
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]+ K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
 
    res = K[n][W]
     
    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i - 1][w]:
            continue
        else:
            chosen_weights.append(val[i - 1])
            res = res - val[i - 1]
            w = w - wt[i - 1]

    chosen_weights.reverse()
    return(chosen_weights)
#####

store_fee_values = []

# Multiple Knapsack loop
for k in range(42):
    size = numpy_fee_split[k]
    weight = numpy_weight_split[k]
    capacity = 95238

    for i in printknapSack(capacity, weight, size, len(size)):
        store_fee_values.append(i)
        # Thus, store_fee_values hold the final set of fee values, whose transaction should be carried out!


''' STEP-5 : Displaying the final output '''
fh = open('block.txt', 'w+')
c = 0
store_weight_values = []
for rows in bitcoin_df.iterrows():
    if rows[1][1] == store_fee_values[c]:
    # Iterating over the rows and if the fee value matches that in the store_fee_values array, the transaction-id is written in the file.
    # The counter 'c' takes care that no two entries are repeated. 
        fh.write(rows[1][0] + '\n')
        store_weight_values.append(rows[1][2])
        c += 1

fh.close()

print('\n')
print('Maximum possible miner fee:', sum(store_fee_values),'satoshis')
print('Total number of transactions to be included:', len(store_fee_values))
print('Final total weight being carried in the block:', sum(store_weight_values), '\n')
print('`block.txt` file avaiable in the current working directory.')