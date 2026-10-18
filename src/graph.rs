use std::{error::Error, fs, path::PathBuf};
// use show_image::{create_window, ImageView, ImageInfo};
use plotters::{prelude::*, style::FontStyle};

use crate::Bar;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const BACKGROUND: RGBColor = RGBColor(13, 18, 30);
const PANEL: RGBColor = RGBColor(23, 30, 46);
const TEXT: RGBColor = RGBColor(232, 238, 247);
const MUTED: RGBColor = RGBColor(147, 162, 184);
const UP: RGBColor = RGBColor(56, 208, 160);
const DOWN: RGBColor = RGBColor(255, 117, 103);

pub(super) fn render_passport(
    token: &str,
    bars: &[Bar],
    features: &[f64],
    minima: &[f64],
    maxima: &[f64],
    prediction: f64,
    raw_return: f64,
    position: f64,
    sharpe: f64,
    equation: &str,
    gradient: Option<&[f64]>,
) -> Result<PathBuf, Box<dyn Error>> {
    if bars.len() < 2 || features.len() != 11 || minima.len() != 11 || maxima.len() != 11 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "market passport requires at least two candles and 11 feature bounds",
        )
        .into());
    }
    if gradient.is_some_and(|values| {
        values.len() != features.len() || values.iter().any(|value| !value.is_finite())
    }) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "market passport gradient must contain 11 finite values",
        )
        .into());
    }

    fs::create_dir_all("graphs")?;
    let path = PathBuf::from("graphs").join(format!("{}_passport.png", safe_filename(token)));
    {
        let root = BitMapBackend::new(&path, (WIDTH, HEIGHT)).into_drawing_area();
        root.fill(&BACKGROUND)?;

        let (header, content) = root.split_vertically(120);
        header.draw(&Text::new(
            format!("MARKET PASSPORT  /  {token}"),
            (42, 48),
            ("sans-serif", 48)
                .into_font()
                .style(FontStyle::Bold)
                .color(&TEXT),
        ))?;
        header.draw(&Text::new(
            "FALSE-PROMISE BY FLMXN  |  RESEARCH DIAGNOSTICS",
            (44, 90),
            ("sans-serif", 24).into_font().color(&MUTED),
        ))?;

        let (chart_area, details_area) = content.split_vertically(520);
        draw_candles(&chart_area, bars)?;
        let (summary_area, features_area) = details_area.split_horizontally(560);
        draw_summary(
            &summary_area,
            prediction,
            raw_return,
            position,
            sharpe,
            equation,
        )?;
        draw_gradient(&features_area, gradient)?;

        root.present()?;
    }
    Ok(path)
}

fn draw_candles(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    bars: &[Bar],
) -> Result<(), Box<dyn Error>> {
    area.fill(&PANEL)?;
    let candles = &bars[bars.len().saturating_sub(80)..];
    let min_price = candles
        .iter()
        .map(|bar| bar.low)
        .fold(f64::INFINITY, f64::min);
    let max_price = candles
        .iter()
        .map(|bar| bar.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((max_price - min_price) * 0.08).max(max_price.abs() * 0.001);
    let x_start = candles.first().expect("at least two candles").ts;
    let x_end = candles.last().expect("at least two candles").ts;
    let mut chart = ChartBuilder::on(area)
        .margin(24)
        .caption(
            "RECENT COMPLETED CANDLES",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .x_label_area_size(52)
        .y_label_area_size(150)
        .build_cartesian_2d(x_start..x_end, (min_price - padding)..(max_price + padding))?;

    chart
        .configure_mesh()
        .x_labels(8)
        .y_labels(7)
        .x_label_formatter(&|timestamp| {
            chrono::DateTime::<chrono::Utc>::from_timestamp(*timestamp, 0)
                .map(|date| date.format("%m-%d %H:%M").to_string())
                .unwrap_or_default()
        })
        .label_style(("sans-serif", 24).into_font().color(&MUTED))
        .axis_style(MUTED.mix(0.45))
        .light_line_style(RGBColor(54, 65, 83).mix(0.35))
        .draw()?;

    let candle_width = (area.dim_in_pixel().0 as f64 / candles.len() as f64 * 0.55)
        .round()
        .clamp(5.0, 16.0) as u32;
    chart.draw_series(candles.iter().map(|bar| {
        CandleStick::new(
            bar.ts,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            UP.filled(),
            DOWN.filled(),
            candle_width,
        )
    }))?;
    Ok(())
}

fn draw_summary(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    prediction: f64,
    raw_return: f64,
    position: f64,
    sharpe: f64,
    equation: &str,
) -> Result<(), Box<dyn Error>> {
    area.fill(&PANEL)?;
    let position_color = if position >= 0.0 { UP } else { DOWN };
    area.draw(&Text::new(
        "MODEL SNAPSHOT",
        (22, 30),
        ("sans-serif", 48)
            .into_font()
            .style(FontStyle::Bold)
            .color(&TEXT),
    ))?;
    area.draw(&Text::new(
        format!("{:+.3}", position),
        (22, 82),
        ("sans-serif", 48)
            .into_font()
            .style(FontStyle::Bold)
            .color(&position_color),
    ))?;
    area.draw(&Text::new(
        format!("POSITION  |  {:+.1}%", position * 100.0),
        (24, 124),
        ("sans-serif", 28).into_font().color(&MUTED),
    ))?;
    area.draw(&Text::new(
        format!("Estimated return  {:+.3}%", raw_return * 100.0),
        (22, 160),
        ("sans-serif", 32).into_font().color(&TEXT),
    ))?;
    area.draw(&Text::new(
        format!("Scaled prediction  {:+.4}", prediction),
        (22, 200),
        ("sans-serif", 28).into_font().color(&MUTED),
    ))?;
    area.draw(&Text::new(
        format!("In-sample Sharpe  {:+.3}", sharpe),
        (22, 225),
        ("sans-serif", 28).into_font().color(&MUTED),
    ))?;
    area.draw(&Text::new(
        "DISCOVERED EQUATION",
        (22, 265),
        ("sans-serif", 28)
            .into_font()
            .style(FontStyle::Bold)
            .color(&TEXT),
    ))?;

    for (index, line) in wrap_text(equation, 42, 4).iter().enumerate() {
        area.draw(&Text::new(
            line.clone(),
            (22, 300 + index as i32 * 30),
            ("sans-serif", 32).into_font().color(&MUTED),
        ))?;
    }
    Ok(())
}

fn draw_gradient(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    gradient: Option<&[f64]>,
) -> Result<(), Box<dyn Error>> {
    const NAMES: [&str; 11] = [
        "Close / open return",
        "Opening gap",
        "High-low range",
        "Upper wick",
        "Lower wick",
        "Relative volume",
        "5-bar mean return",
        "5-bar volatility",
        "vs. 50-bar average",
        "vs. 200-bar average",
        "Time of day",
    ];
    area.fill(&PANEL)?;
    area.draw(&Text::new(
        "LOCAL GRADIENT  /  ∂prediction / ∂x_scaled",
        (32, 30),
        ("sans-serif", 48)
            .into_font()
            .style(FontStyle::Bold)
            .color(&TEXT),
    ))?;

    let Some(gradient) = gradient else {
        area.draw(&Text::new(
            "Unavailable: equation derivative could not be evaluated.",
            (32, 90),
            ("sans-serif", 48).into_font().color(&MUTED),
        ))?;
        return Ok(());
    };

    let max_abs = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let panel_width = area.dim_in_pixel().0 as i32;
    let center_x = panel_width * 2 / 3;
    let half_width = (panel_width / 4).max(1);
    let value_x = panel_width - 115;
    for (index, value) in gradient.iter().enumerate() {
        let y = 88 + index as i32 * 30;
        let magnitude = if max_abs == 0.0 {
            0.0
        } else {
            (value.abs() / max_abs).clamp(0.0, 1.0)
        };
        area.draw(&Text::new(
            format!("x{}  {}", index + 1, NAMES[index]),
            (32, y + 4),
            ("sans-serif", 32).into_font().color(&TEXT),
        ))?;
        area.draw(&Text::new(
            format!("{:+.3}", value),
            (value_x, y + 4),
            ("sans-serif", 32).into_font().color(&MUTED),
        ))?;
        area.draw(&PathElement::new(
            vec![(center_x, y - 5), (center_x, y + 17)],
            MUTED.mix(0.55),
        ))?;

        let length = (magnitude * half_width as f64).round() as i32;
        if length > 0 {
            let (left, right) = if *value >= 0.0 {
                (center_x, center_x + length)
            } else {
                (center_x - length, center_x)
            };
            let color = if *value >= 0.0 { UP } else { DOWN };
            area.draw(&Rectangle::new(
                [(left, y), (right, y + 12)],
                color.mix(0.88).filled(),
            ))?;
        }
    }
    Ok(())
}

fn safe_filename(token: &str) -> String {
    let name: String = token
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if name.is_empty() {
        "asset".into()
    } else {
        name
    }
}

fn wrap_text(text: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut lines: Vec<String> = characters
        .chunks(max_chars)
        .take(max_lines)
        .map(|chunk| chunk.iter().collect())
        .collect();
    if characters.len() > max_chars * max_lines {
        if let Some(last) = lines.last_mut() {
            last.truncate(max_chars.saturating_sub(3));
            last.push_str("...");
        }
    }
    lines
}

// #[show_image::main]
// fn show(path: &str) -> Result<(), Box<dyn std::error::Error>> {
//     let img = image::open(path)?;
//     let window = create_window(path, Default::default())?;
//     window.set_image("my-image", img)?;
//     window.run()?;
//     Ok(())
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_png_market_passport() {
        let bars: Vec<Bar> = (0..4)
            .map(|index| {
                let price = 100.0 + index as f64;
                Bar {
                    ts: 1_700_000_000 + index * 3600,
                    open: price,
                    high: price + 2.0,
                    low: price - 1.0,
                    close: price + 1.0,
                    volume: 1000.0,
                }
            })
            .collect();
        let token = format!("graph-test-{}", std::process::id());
        let path = render_passport(
            &token,
            &bars,
            &[0.0; 11],
            &[-1.0; 11],
            &[1.0; 11],
            0.25,
            0.01,
            0.5,
            1.2,
            "tanh(x1 + x2)",
            Some(&[-0.897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0]),
        )
        .expect("passport should render");

        let png = fs::read(&path).expect("rendered image should exist");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), WIDTH);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), HEIGHT);
        fs::remove_file(path).expect("test image should be removable");
    }
}
