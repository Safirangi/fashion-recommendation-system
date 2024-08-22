
def male_size_chart(shoulder_length, waist):
    # Determine top size
    if shoulder_length < 38:
        top_size = "XS"
    elif 38 <= shoulder_length < 40:
        top_size = "S"
    elif 40 <= shoulder_length < 42:
        top_size = "M"
    elif 42 <= shoulder_length < 44:
        top_size = "L"
    elif 44 <= shoulder_length < 46:
        top_size = "XL"
    elif 46 <= shoulder_length < 48:
        top_size = "XXL"
    else:
        top_size = "XXXL"

    # Determine bottom size
    if waist < 71.12:
        bottom_size = "28"
    elif 71.12 <= waist < 76.2:
        bottom_size = "30"
    elif 76.2 <= waist < 81.28:
        bottom_size = "32"
    elif 81.28 <= waist < 86.36:
        bottom_size = "34"
    elif 86.36 <= waist < 91.44:
        bottom_size = "36"
    else:
        bottom_size = "38"

    return top_size, bottom_size


def female_size_chart(shoulder_length, waist):
    # Determine top size
    if shoulder_length < 33:
        top_size = "XS"
    elif 33 <= shoulder_length < 35:
        top_size = "S"
    elif 35 <= shoulder_length < 37:
        top_size = "M"
    elif 37 <= shoulder_length < 39:
        top_size = "L"
    elif 39 <= shoulder_length < 41:
        top_size = "XL"
    elif 41 <= shoulder_length < 43:
        top_size = "XXL"
    else:
        top_size = "XXXL"

     # Determine bottom size
    if waist < 71.12:
        bottom_size = "28"
    elif 71.12 <= waist < 76.2:
        bottom_size = "30"
    elif 76.2 <= waist < 81.28:
        bottom_size = "32"
    elif 81.28 <= waist < 86.36:
        bottom_size = "34"
    elif 86.36 <= waist < 91.44:
        bottom_size = "36"
    else:
        bottom_size = "38"

    return top_size, bottom_size
