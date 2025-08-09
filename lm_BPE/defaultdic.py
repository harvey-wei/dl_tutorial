from collections import defaultdict


# When to use defaultdict
# Use it when:
#
# You want to avoid writing if key not in dict: dict[key] = ...
#
# You're accumulating, grouping, or counting

freq = defaultdict(int)  # int() → 0
words = ['a', 'b', 'a', 'c', 'b', 'a']

for word in words:
    freq[word] += 1

print(freq)  # {'a': 3, 'b': 2, 'c': 1}
