import nltk
from preprocess import clean_str


max_length = 0
max_sentence = ""

MIN_LENGTH = 10
LIMIT = 59
DEV_SIZE = 0.1
TEST_SIZE = 0.1

duplicates = 0
sentences = 0

header = True
skipped = 0
skipped_duplicates = 0
sentences = 0

lines = []
training_data = []  # List to store training data tuples
test_data = []      # List to store test data tuples

with open('./data/quora_duplicate_questions.tsv') as f:
    for line in f:
        if header:
            header = False
            continue

        print(line)
        elements = line.strip('\n').split('\t')

        q1 = clean_str(elements[3])  # Assuming clean_str is a function to clean the question strings
        q2 = clean_str(elements[4])
        duplicate = elements[5]

        q1_length = len(q1.split())
        q2_length = len(q2.split())

        if q1_length > LIMIT or q2_length > LIMIT:
            skipped += 1
            if duplicate == '1':
                skipped_duplicates += 1
            continue

        if q1_length + q2_length < MIN_LENGTH:
            skipped += 1
            if duplicate == '1':
                skipped_duplicates += 1
            continue

        if q1_length > max_length:
            max_length = q1_length
            max_sentence = q1

        if q2_length > max_length:
            max_length = q2_length
            max_sentence = q2

        if duplicate == '1':
            duplicates += 1

        # If a question is empty, replace it with a placeholder (".")
        if len(q1) == 0:
            q1 = "."
        if len(q2) == 0:
            q2 = "."

        # Append the tuple (q1, q2, duplicate) to the lines list
        lines.append((q1, q2, duplicate))

        # print(elements)
        sentences += 1

# Splitting the data into training and test sets
test_index = -1 * int(len(lines) * TEST_SIZE)
training_lines = lines[:test_index]
test_lines = lines[test_index:]

# Saving training and test data as lists of tuples
for line in training_lines:
    q1, q2, duplicate = line
    training_data.append((q1, q2, int(duplicate)))  # Convert duplicate to integer

for line in test_lines:
    q1, q2, duplicate = line
    test_data.append((q1, q2, int(duplicate)))  # Convert duplicate to integer

# Writing the data to separate files
with open('./data/train.full.tsv', 'w') as fw:
    for line in training_data:
        q1, q2, duplicate = line
        fw.write('%s\t%s\t%s\n' % (duplicate, q1, q2))

with open('./test.full.tsv', 'w') as fw_test:
    for line in test_data:
        q1, q2, duplicate = line
        fw_test.write('%s\t%s\t%s\n' % (duplicate, q1, q2))

# Printing some statistics
print("Training data size: ", len(training_data))
print("Test data size: ", len(test_data))
print("Longest sentence: %s (%d)" % (max_sentence, max_length))
print("Duplicates: %d (%.2f%%)" % (duplicates, (1.0 * duplicates) / sentences * 100))
print("Skipped: %d (%d skipped duplicates)" % (skipped, skipped_duplicates))
