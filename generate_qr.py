
import qrcode  # pyright: ignore[reportMissingModuleSource]

# Your GitHub repository link
github_link = "https://github.com/Keerthana-Kurmilla/Explainable-Acute-Lymphoblastic-Leukemia-Classification-from-Blood-Smear-Images"

# Create QR code
qr = qrcode.QRCode(
    version=None,
    # ERROR_CORRECT_H = 2 (use integer literal to avoid type-checker issues
    # with some qrcode package installations that do not expose `constants`)
    error_correction=2,
    box_size=15,
    border=4
)

qr.add_data(github_link)
qr.make(fit=True)

# Generate QR image
img = qr.make_image(
    fill_color="black",
    back_color="white"
)

# Save QR code
with open("Capstone_Project_QR.png", "wb") as qr_file:
    img.save(qr_file)

print("QR Code generated successfully!")
print("File: Capstone_Project_QR.png")
