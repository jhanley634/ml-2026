mod mirror;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello world!");
    mirror::mirror()?;

    Ok(())
}
