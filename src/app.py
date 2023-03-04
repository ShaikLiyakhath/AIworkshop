from flask import Flask, request, jsonify

from mynn import MyFirstNN


app=Flask(__name__)
N_N = MyFirstNN()

@app.route('/api/myfisrtnn', methods=['GET', 'POST'])
def myneuralnetwork():
    if request.method == 'GET':
        return jsonify('This is own trained network')
    else:
        input_json = request.json
        input_arr = input_json['input']
        output = N_N.predict(input_arr)
        return jsonify(output)

if __name__ == '__main__':
    app.run('0.0.0.0', port=8416)