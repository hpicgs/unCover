# Print iterations progress
def printProgressBar(iteration, total, decimals=1, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = fill * filledLength + '-' * (100 - filledLength)
    print(f'\r |{bar}| {percent}% ', end=printEnd)
    if iteration == total:
        print()