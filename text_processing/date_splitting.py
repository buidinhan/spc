def split_date(string):
    """'dd/mm/yy' -> (dd, mm, yyyy)"""
    try:
        portions = string.split("/")

        day = int(portions[0])
        month = int(portions[1])
        if int(portions[2]) < 100:
            year = int(portions[2]) + 2000
        else:
            year = int(portions[2])

        return day, month, year

    except:
        return string, "", ""


def read_data(path):
    dates = []
    with open(path, "r") as f:
        for line in f:
            dates.append(line.strip())
    return dates


def output(dates, path):
    with open(path, "w") as f:
        f.write("original,day,month,year\n")
        for date in dates:
            day, month, year = split_date(date)
            f.write("{},{},{},{}\n".format(date, day, month, year))


def process(source, destination):
    dates = read_data(source)
    output(dates, destination)


if __name__ == "__main__":
    source = "input.txt"
    destination = "output.csv"
    process(source, destination)
