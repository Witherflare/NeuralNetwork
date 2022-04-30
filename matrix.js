class Matrix {
  constructor (rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = [];
    for (let i = 0; i < this.rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = 0;
      }
    }
  }

  static multiply (a, b) {
    // Matrix product
    if (a.cols !== b.rows) {
      throw new Error('Columns of A must match rows of B.');
    }
    let result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        // Dot product of values in col
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  multiply (n) {
    // Scalar product
    let result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] * n;
      }
    }
    return result;
  }

  // Either adds a scalar to a matrix or does element-wise addition (adds a matrix to a matrix)
  add (n) {
    if (n instanceof Matrix) {
      let result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] + n.data[i][j];
        }
      }
      return result;
    } else {
      let result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] + n;
        }
      }
      return result;
    }
  }

  randomize () {
    let result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = Math.floor(Math.random() * 10)
      }
    }
    return result;
  }

  transpose () {
    let result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j][i] = this.data[i][j];
      }
    }
    return result;
  }

  // map applies a function to every element of the matrix
  map (func) {
    let result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = func(this.data[i][j]);
      }
    }
    return result;
  }

  // print the matrix
  print () {
    console.table(this.data);
  }
}

let a = new Matrix(3, 2).randomize()
let b = new Matrix(2, 2).randomize()
a.print()
b.print()
Matrix.multiply(a, b).print()