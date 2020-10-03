from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    prediction = 'Value from prediction'

    return render_template('index.html', prediction=prediction)


@app.route('/info')
def home():
    return render_template('info.html')


@app.route('/prediction')
def prediction():

    return render_template('prediction.html')


@app.route('/signup_form')
def signup_form():

    return render_template('signup_form.html')

@app.route('/report')
def report():

    lower_letter = False
    upper_letter = False
    last_char = False

    username = request.args.get('username')

    lower_letter = any(c.islower() for c in username)
    upper_letter = any(c.isupper() for c in username)
    last_char = username[-1].isdigit()

    report = lower_letter and upper_letter and last_char

    return render_template('report.html', report=report, lower=lower_letter, upper=upper_letter, last_char=last_char)
    

@app.route('/thank_you')
def thank_you():
    first = request.args.get('first')
    last = request.args.get('last')

    return render_template('thank_you.html', first=first, last=last)



@app.errorhandler(404)
def page_not_found(e):

    return render_template('404.html'), 404


@app.route('/ai/<name>')
def name(name):

    if (name[-1] == 'd'):
        name = 'name '+'k-means'
    else:
        name = 'name '+ 'knn'

    page = '<h1>Artificial Intelligence {}</h1>'.format(name)
    return page

if __name__ == "__main__":
    app.run()