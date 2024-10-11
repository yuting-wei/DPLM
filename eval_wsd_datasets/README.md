
# Data Format Description

Data is stored in a pkl file with the following format:

```python
{
    'word': word,              # Word
    'word_id': word_id,        # Word ID
    'sense_id': sense_id,      # Sense ID
    'examples': [[train_examples, dynasty], ...]  # Examples and corresponding historical period
}
```

## Historical Period Classification

### Classified into 3 historical periods

| Dynasty        | ID |
| -------------- | -- |
| Ancient        | 1  |
| Middle Ancient | 2  |
| Near Ancient   | 3  |

### Classified into 8 historical periods

| Dynasty                                      | ID |
| -------------------------------------------- | -- |
| Zhou                                         | 8  |
| Spring and Autumn period and Warring States  | 7  |
| Han                                          | 6  |
| Wei, Jin, Southern and Northern Dynasties    | 5  |
| Sui, Tang and Five Dynasties                 | 4  |
| Song                                         | 3  |
| Yuan and Ming                                | 2  |
| Qing                                         | 1  |

## Note

- For the classification of 3 historical periods, corrections are made in the `wsd_model.py` file through the following dictionary:
    ```python
    dynastys_dict = {'1': '3', '2': '2', '3': '1'}
    ```
