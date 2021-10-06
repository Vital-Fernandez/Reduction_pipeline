def review_filepath(file_list, character = ['+', 'plus']):

    for i, file_path in enumerate(file_list):
        if character[0] in file_path:
            file_list[i] = file_path.replace(character[0], character[1])

    return file_list


