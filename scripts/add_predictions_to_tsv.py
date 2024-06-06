import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# Get the command line arguments
binned_exon_sequences_path = sys.argv[1]
pred_results_path = sys.argv[2]
rbp = sys.argv[3]

# Load the binned exon sequences
binned_exon_sequences_df = pd.read_csv(binned_exon_sequences_path, sep='\t')

# Load the prediction results
pred_results = np.load(pred_results_path)

# Plot the distribution of prediction probabilities
plt.hist(pred_results, bins=50)
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.savefig('prediction_probabilities_distribution.png')
plt.show()

# Add the prediction results as a new column in the dataframe
binned_exon_sequences_df[rbp] = pred_results

# Save the updated dataframe to a new TSV file
output_path = binned_exon_sequences_path
binned_exon_sequences_df.to_csv(output_path, sep='\t', index=False)

print(f"Added predictions to {output_path}")
