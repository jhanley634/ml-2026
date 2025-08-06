use opencv::core::{AlgorithmHint, Mat, Point, Scalar, Size, flip};
use opencv::highgui::{destroy_all_windows, imshow, wait_key};
use opencv::imgproc::{ColorConversionCodes, HersheyFonts, cvt_color, get_text_size, put_text};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, VideoWriter};

#[derive(Debug)]
pub enum MirrorError {
    CvError(opencv::Error),
}

impl std::error::Error for MirrorError {}

impl std::fmt::Display for MirrorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MirrorError::CvError(err) => write!(f, "CvError: {err}"),
        }
    }
}

// for the `?` operator
impl From<opencv::Error> for MirrorError {
    fn from(err: opencv::Error) -> Self {
        MirrorError::CvError(err)
    }
}

pub struct Fps {
    frame_count: u64,
    start_time: std::time::Instant,
}

impl Fps {
    pub fn new() -> Self {
        Fps {
            frame_count: 0,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn update(&mut self) {
        self.frame_count += 1;
    }

    pub fn calculate_fps(&self) -> f32 {
        let duration = self.start_time.elapsed();
        if duration.as_secs_f32() == 0.0 {
            return 0.0;
        }
        (self.frame_count as f32) / duration.as_secs_f32()
    }

    pub fn get_text(&self, font: i32, color: Scalar, thickness: f32) -> String {
        let fps = self.calculate_fps();
        format!("FPS: {fps:.1}")
    }
}

pub fn mirror() -> Result<(), MirrorError> {
    println!("Hit 'q' to quit");

    let mut cap = VideoCapture::new(0, 0)?;

    let frame_width = 640;
    let frame_height = 480;

    let mut writer = VideoWriter::new(
        "/tmp/output.avi",
        videoio::VideoWriter::fourcc('M', 'P', 'E', 'G')?,
        10.0,
        Size::new(frame_width, frame_height),
        false,
    )?;

    let font = HersheyFonts::FONT_HERSHEY_SIMPLEX as i32;
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let thickness: f32 = 2.0;

    let mut fps_tracker = Fps::new();

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

        let text = fps_tracker.get_text(font, color, thickness);
        let mut base_line = 0;
        let text_size = get_text_size(&text, font, 1.0, 0, &mut base_line).unwrap();
        let text_origin = (
            flipped_frame.cols() - text_size.width as i32 - 10,
            flipped_frame.rows() - text_size.height as i32 - 10,
        );
        put_text(
            &mut flipped_frame,
            &text,
            Point::new(text_origin.0, text_origin.1),
            font,
            1.0,
            color,
            thickness.round() as i32,
            font,
            false,
        )?;

        imshow("Mirror", &flipped_frame)?;

        let key = wait_key(1)?;
        if key == 'q' as i32 {
            break;
        }

        fps_tracker.update();
        writer.write(&flipped_frame)?;
    }

    cap.release()?;
    destroy_all_windows()?;
    writer.release()?;

    Ok(())
}
