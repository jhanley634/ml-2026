use opencv::{
    core::{AlgorithmHint, Mat, flip},
    highgui::{destroy_all_windows, imshow, wait_key},
    imgproc::{ColorConversionCodes, cvt_color},
    prelude::*,
    videoio::VideoCapture,
};

fn mirror() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hit 'q' to quit");

    let mut cap = VideoCapture::new(0, 0)?;

    loop {
        let mut frame = Mat::default();
        let ret = cap.read(&mut frame)?;
        if !ret {
            break;
        }

        // Convert to grayscale
        let mut gray = Mat::default();
        cvt_color(
            &frame,
            &mut gray,
            ColorConversionCodes::COLOR_BGR2GRAY as i32,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut flipped_frame = Mat::default();
        flip(&gray, &mut flipped_frame, 1)?; // horizontally

        imshow("Mirror", &flipped_frame)?;

        let key = wait_key(1)?;
        if key == 'q' as i32 {
            break;
        }
    }

    cap.release()?;
    destroy_all_windows()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello world!");
    mirror()?;

    Ok(())
}
