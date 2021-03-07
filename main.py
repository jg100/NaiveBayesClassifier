
if __name__ == '__main__':

    my_dataset = {  # data set taken from MarketWatch.com headlines.
        "Stock market crash? No, but rising bond yields are sparking a nerve-racking rotation below the surface":
            "Stock Related",
        "Cuomo advisers got health officials to alter report on COVID-19 nursing-home deaths": "Not Stock Related",
        "Broadcom Stock Drops Despite Solid Earnings": "Stock Related",
        "Companies Are Issuing Guidance Again. What Investors Should Know.": "Stock Related",
        "Mortgage rates soar above 3% — how high can they go before they scare off homebuyers?": "Not Stock Related",
        "What’s worth streaming in March 2021: It’s time to watch some dumb but fun stuff": "Not Stock Related",
        "EBay will delist ‘banned’ Dr. Seuss books being resold for thousands of dollars": "Not Stock Related",
        "What happens if I don't file my taxes?": "Not Stock Related",
        "Dow tumbles for second day, U.S. stocks swoon as bond yields flirt with year’s highs": "Stock Related",
        "Dow finishes 144 points lower after best one-day rally in months": "Stock Related",
        "As rising Treasury yields spook stock investors, March looms like a lion": "Stock Related",
        "U.S. COVID death toll above 518,000 as Biden adds voice to outrage at Texas and Mississippi for reopening":
            "Not Stock Related",
        "Single dose of AstraZeneca or Pfizer COVID-19 vaccine cuts hospitalization risk by more than 80%, study shows":
            "Not Stock Related ",
        "Asian-American businesses are dealing with two viruses: Reeling from racist incidents, many are "
        "hurting financially during COVID-19": "Not Stock Related",
        "CDC: Double masking is more likely to protect against the coronavirus": "Not Stock Related",
        "Only 50 people are known to have contracted COVID-19 more than once — "
        "but new strains have medical experts on high alert": "Not Stock Related"
    }

    # Calculate prior
    dataset_size = len(my_dataset)

    # P(Stock related)
    stock_counter_class = 0
    not_stock_counter_class = 0
    words_in_stock_related = 0
    words_in_not_stock_related = 0

    # Splitting the data set into stock related and not stock related
    unique_words_stock = {}
    unique_words_not_stock = {}
    for i in my_dataset.keys():
        split_string = i.lower().split()
        if my_dataset.get(i) == "Stock Related":
            # Counting the class
            stock_counter_class += 1
            # Counting the words in the string/class
            for j in split_string:
                unique_words_stock[j] = 0

            words_in_stock_related += len(split_string)
        else:
            # Adding one to class count for Not Stock
            not_stock_counter_class += 1
            # Counting the number of words present in the class
            for j in split_string:
                unique_words_not_stock[j] = 0
            words_in_not_stock_related += len(split_string)

    # Probability a document is part of a class (priors)
    p_stock_related = stock_counter_class / dataset_size
    p_not_stock_related = not_stock_counter_class / dataset_size

    # Counting each unique word in the dataset and incrementing the
    # corresponding dictionary created on line 38 and 39 (priors)
    # Using laplace smoothing to take care of any 0 occurences
    for i in my_dataset:
        split_string = i.lower().split()
        if my_dataset.get(i) == "Stock Related":
            for word in split_string:
                if word in unique_words_stock:
                    unique_words_stock[word] += 1
        else:
            for word in split_string:
                if word in unique_words_not_stock:
                    unique_words_not_stock[word] += 1

    # print("--- Priors ---")
    # print("Stock Related: " + str(p_stock_related))
    # print("Not Stock Related: " + str(p_not_stock_related) + "\n")

    # print(unique_words_stock)
    # print(unique_words_not_stock)

    # Calculate conditional probs

    # List of all the unique words in all classes
    all_unique_words = list(unique_words_stock.keys()) + list(unique_words_not_stock.keys())
    total_unique_words = len(unique_words_not_stock.keys()) + len(unique_words_stock.keys())

    prob_map = {}
    # Calcualting the prob of each word in each class and adding it to dictionary
    for x in all_unique_words:
        # print(unique_words_stock.get(x))
        p_x_given_stock = 0
        p_x_given_not_stock = 0

        if x in unique_words_stock.keys():
            p_x_given_stock = (unique_words_stock.get(x) + 1) / (words_in_stock_related + total_unique_words)
            p_x_given_not_stock = 1 / (words_in_not_stock_related + total_unique_words)

        elif x in unique_words_not_stock.keys():
            p_x_given_stock = 1 / (words_in_stock_related + total_unique_words)
            p_x_given_not_stock = (unique_words_not_stock.get(x) + 1) / (
                        words_in_not_stock_related + total_unique_words)

        # word = [P(word | stock related class), P(word | not stock related class) ]
        # index 0 is the prob of words given stock docs, index 1 is prob word given not stock related docs
        # Incorporated laplace smoothing as well.
        prob_map[x] = [p_x_given_stock, p_x_given_not_stock]

    # print(prob_map)

    user_input = input("Enter a sentence: ")
    user_input = user_input.lower()
    tokenized_input = user_input.split()

    # Calculate the probability of stock related INPUT and not stock related input
    # Use matrix to pull if value is in class, if not use lapalce smoothing to calc prob.
    prob_stock_input = 1
    prob_not_stock_input = 1
    for x in tokenized_input:
        if x in prob_map:
            prob_stock_input *= prob_map[x][0]
            prob_not_stock_input *= prob_map[x][1]
        else:
            prob_stock_input *= 1 / (words_in_stock_related + total_unique_words)
            prob_not_stock_input *= 1 / (words_in_not_stock_related + total_unique_words)

    prob_stock_input *= p_stock_related
    prob_not_stock_input *= p_not_stock_related

    #Output
    print("Input text:\n" + user_input)

    print("Output:\nClassified as Stock Related input") if prob_stock_input > prob_not_stock_input else print(
        "Classified as "
        "not Stock "
        "related input")

    print("Explanation:\n Class 0: Stock related\nClass 1: Not stock related")
