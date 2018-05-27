import configparser



def read_section(file_name, section):

    config = configparser.ConfigParser()

    config.read(file_name)
    dictionary = {}

    for option in config.options(section):

        dictionary[option] = config.get(section, option)

    return dictionary





def main():


    print(read_section("part1.ini", "part1"))





if __name__ == "__main__":

    main()
