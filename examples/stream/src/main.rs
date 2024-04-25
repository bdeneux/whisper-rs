use samplerate::ConverterType;
use samplerate::Samplerate;
use cpal::FromSample;
use cpal::Sample;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Mutex;
use std::sync::Arc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavSpec;
use ringbuf::{LocalRb, Rb, SharedRb};
use std::time::{Duration, Instant};
use std::{cmp, thread};
use whisper_rs::WhisperError;

const LATENCY_MS: f32 = 5000.0;

fn main() -> Result<(), &'static str> {

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("failed to get default input device");

    println!("Using default input device: \"{}\"", input_device.name().expect("failed get input device name"));

    let config = input_device.default_input_config().expect("failed get default input config");
    let latency_frames = (LATENCY_MS / 1_000.0) * config.sample_rate().0 as f32;
    let latency_samples = latency_frames as usize * config.channels() as usize;
    let sampling_freq = config.sample_rate().0 as f32 / 2.0;

    let ring = SharedRb::new(latency_samples * 2);
    let (mut producer, mut consumer) = ring.split();

    // Setup microphone callback
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut output_fell_behind = false;
        for &sample in data {
            if producer.push(sample).is_err() {
                output_fell_behind = true;
            }
        }
        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    let input_stream = input_device.build_input_stream(&config.config(), input_data_fn, err_fn, None).expect("failed build input stream");
    println!("starting recording microphone");
    input_stream.play().expect("failed start input stream");

    // Remove the initial samples
    consumer.pop_iter().count();

    let mut start_time = Instant::now();
    let mut loop_num = 0;
    let writer = hound::WavWriter::create("speech.wav", wav_spec_from_config(&config)).expect("failed create wav writter");
    let writer = Arc::new(Mutex::new(Some(writer)));

    loop {
        loop_num += 1;

        // Only run every LATENCY_MS
        let duration = start_time.elapsed();
        let latency = Duration::from_millis(LATENCY_MS as u64);
        if duration < latency {
            let sleep_time = latency - duration;
            thread::sleep(sleep_time);
        } else {
            panic!("Classification got behind. It took to long. Try using a smaller model and/or more threads");
        }
        start_time = Instant::now();

        // Collect the samples

        let samples: Vec<_> = consumer.pop_iter().collect();
        let samples = whisper_rs::convert_stereo_to_mono_audio(&samples).unwrap();
        let samples = resample(config.sample_rate().0, 16000, 1, &samples);

        write_input_data::<f32, f32>(&samples, &writer);


        writer.lock().unwrap().take().unwrap().finalize().expect("finalize");
        println!("end")
    }

    Ok(())
}

pub fn resample(from_rate: u32, to_rate: u32, channel: usize, samples: &[f32]) -> Vec<f32> {
    let mut converter = Samplerate::new(ConverterType::SincBestQuality, from_rate, to_rate, channel).unwrap();
    converter.process_last(samples).expect("failed resample")
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
    where
        T: Sample,
        U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}