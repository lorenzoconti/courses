import turicreate as tc
import matplotlib.pyplot as plt

loans = tc.SFrame.read_csv('lending-club-data.csv')

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
            ]

loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print('Dropping %s observations; keeping %s '.format(num_rows_with_na, num_rows))

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print('Percentage of safe loans                 : {}'.format( len(safe_loans) / float(len(loans_data))))
print('Percentage of risky loans                : {}'.format(len(risky_loans) / float(len(loans_data))))
print('Total number of loans in our new dataset : {}'.format(len(loans_data)))

train_data, validation_data = loans_data.random_split(.8, seed=1)

# gradient boosted tree classifier
model = tc.boosted_trees_classifier.create(train_data, validation_set=None, target=target, features=features,
                                           max_iterations=5)

# make predictions
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data = validation_risky_loans[0:2].append(validation_safe_loans[0:2])

print('predictions: {}'.format(model.predict(sample_validation_data)))
print('probabilities: {}'.format(model.predict(sample_validation_data, output_type='probability')))

print(model.evaluate(validation_data))

validation_data['predictions'] = model.predict(validation_data, output_type='probability')
validation_data.sort('predictions', ascending=False)

most_positive_loans = validation_data['grade'][0:5]
print(most_positive_loans)

validation_data.sort('predictions', ascending=True)

most_negative_loans = validation_data['grade'][0:5]
print(most_negative_loans)

models = []
train_errors = []
validation_errors = []
max_iterations_list = [10, 50, 100, 200, 500]
for max_iterations in max_iterations_list:
    m = tc.boosted_trees_classifier.create(train_data, validation_set=None, target=target, features=features,
                                           max_iterations=max_iterations, verbose=False)
    models.append(m)
    train_error = m.evaluate(train_data)['accuracy']
    print(train_error)
    train_errors.append(1-train_error)
    validation_error = m.evaluate(validation_data)['accuracy']
    print(validation_error)
    validation_errors.append(1-validation_error)


def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


plt.plot(max_iterations_list, train_errors, label='Training error')
plt.plot(max_iterations_list, validation_errors, label='Validation error')

make_figure(dim=(10, 5), title='Error vs number of trees', xlabel='NUmber of trees', ylabel='Classification error',
            legend='best')