use ndarray::{array, Axis};
use ndarray_shuffle::NdArrayShuffleExt;

fn main() {
    let sorted = array![
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    ];

    let shuffled_axis_0 = sorted.shuffle(Axis(0)).unwrap();
    println!("Shuffled along Axis 0\n{shuffled_axis_0:#?}");
    // [
    //      [[19, 20, 21],  [22, 23, 24],   [25, 26, 27]],
    //      [[1, 2, 3],     [4, 5, 6],      [7, 8, 9]],
    //      [[10, 11, 12],  [13, 14, 15],   [16, 17, 18]],
    // ];

    let shuffled_axis_1 = sorted.shuffle(Axis(1)).unwrap();
    println!("Shuffled along Axis 1\n{shuffled_axis_1:#?}");
    // [
    //      [[7, 8, 9],     [1, 2, 3],      [4, 5, 6]],
    //      [[16, 17, 18],  [10, 11, 12],   [13, 14, 15]],
    //      [[25, 26, 27],  [19, 20, 21],   [22, 23, 24]],
    // ];

    let shuffled_axis_2 = sorted.shuffle(Axis(2)).unwrap();
    println!("Shuffled along Axis 2\n{shuffled_axis_2:#?}");
    // [
    //      [[3, 1, 2],     [6, 4, 5],      [9, 7, 8]],
    //      [[11, 10, 12],  [15, 13, 14],   [17, 16, 18]],
    //      [[21, 19, 20],  [22, 23, 24],   [25, 26, 27]],
    // ];
}
