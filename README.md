# ndarray-shuffle

The `ndarray-shuffle` provides an extension for [ndarray](https://github.com/rust-ndarray/ndarray)
to shuffle elements in-place or into a new array along any axis, preserving order for other axes.

It supports arrays with standard layout only.

## Example

```rust
let sorted = array![
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
];

let shuffled_axis_0 = sorted.shuffle(Axis(0)).unwrap();
println!("{shuffled_axis_0:#?}");
// [
//      [[19, 20, 21],  [22, 23, 24],   [25, 26, 27]],
//      [[1, 2, 3],     [4, 5, 6],      [7, 8, 9]],
//      [[10, 11, 12],  [13, 14, 15],   [16, 17, 18]],
// ];
```

## License
Licensed under either of [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT license](./LICENSE-MIT) at your option.
