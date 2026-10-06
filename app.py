"""Personal Research Assistant Agent.

Extracts text from a PDF file or web page, summarizes it with Gemini, and
saves the result as a Markdown file in the output directory.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader


MODEL_NAME = "gemini-3.8-flash"
OUTPUT_DIR = Path("output")


def extract_from_pdf(file_path: str | Path) -> str:
	"""Extract all readable text from a PDF file."""
	path = Path(file_path)
	if not path.is_file():
		raise FileNotFoundError(f"File PDF tidak ditemukan: {path}")

	reader = PdfReader(str(path))
	pages = [page.extract_text() or "" for page in reader.pages]
	text = "\n\n".join(pages).strip()
	if not text:
		raise ValueError(f"Tidak ada teks yang dapat diekstrak dari PDF: {path}")
	return text


def extract_from_url(url: str) -> str:
	"""Extract the main readable text from a web page."""
	response = requests.get(
		url,
		headers={"User-Agent": "PersonalResearchAssistant/1.0"},
		timeout=30,
	)
	response.raise_for_status()

	soup = BeautifulSoup(response.text, "html.parser")
	for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
		element.decompose()

	main_content = soup.find("main") or soup.find("article") or soup.body or soup
	text = main_content.get_text(" ", strip=True)
	text = re.sub(r"\s+", " ", text).strip()
	if not text:
		raise ValueError(f"Tidak ada teks utama yang dapat diekstrak dari URL: {url}")
	return text


def create_summary_chain() -> ChatPromptTemplate:
	"""Create the LangChain pipeline used to summarize extracted text."""
	load_dotenv()
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"GEMINI_API_KEY belum ditemukan. Tambahkan ke file .env atau environment."
		)

	llm = ChatGoogleGenerativeAI(
		model=MODEL_NAME,
		google_api_key=api_key,
		temperature=0.2,
	)
	prompt = ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"Anda adalah asisten riset yang teliti. Rangkum sumber berikut "
				"dalam bahasa Indonesia menggunakan Markdown yang rapi. "
				"Jangan mengarang fakta yang tidak ada di sumber. Gunakan "
				"struktur persis: Judul / Topik Utama, Ringkasan Eksekutif "
				"(2-3 kalimat), Poin-Poin Kunci & Temuan Utama, Data / "
				"Statistik Penting (jika ada), dan Kesimpulan.",
			),
			("human", "Sumber yang perlu dirangkum:\n\n{text}"),
		]
	)
	return prompt | llm | StrOutputParser()


def summarize_text(text: str) -> str:
	"""Summarize extracted text with the configured Gemini model."""
	chain = create_summary_chain()
	return chain.invoke({"text": text})


def _safe_output_name(source: str) -> str:
	"""Create a filesystem-friendly Markdown filename from the source."""
	parsed = urlparse(source)
	name = Path(parsed.path).stem if parsed.scheme else Path(source).stem
	name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")
	return name or "research-summary"


def save_summary(summary: str, source: str, output_dir: str | Path = OUTPUT_DIR) -> Path:
	"""Save a summary to output_dir and return the created path."""
	destination = Path(output_dir)
	destination.mkdir(parents=True, exist_ok=True)
	output_path = destination / f"{_safe_output_name(source)}.md"
	output_path.write_text(summary.rstrip() + "\n", encoding="utf-8")
	return output_path


def _is_url(value: str) -> bool:
	parsed = urlparse(value)
	return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Rangkum PDF atau halaman web menggunakan Gemini."
	)
	parser.add_argument(
		"source",
		help="URL halaman web atau path ke file PDF",
	)
	parser.add_argument(
		"--output-dir",
		default=str(OUTPUT_DIR),
		help="Folder untuk menyimpan hasil Markdown (default: output)",
	)
	args = parser.parse_args()

	load_dotenv()
	if _is_url(args.source):
		text = extract_from_url(args.source)
	else:
		text = extract_from_pdf(args.source)

	print(' sedang merangkum...')
	summary = summarize_text(text)
	output_path = save_summary(summary, args.source, args.output_dir)
	output_folder = Path(args.output_dir).as_posix().rstrip("/") + "/"
	print(' Selesai! Ringkasan disimpan ke folder output.')
	print(
		f"Konfirmasi: file ringkasan berhasil disimpan ke folder "
		f"{output_folder}: {output_path}"
	)


if __name__ == "__main__":
	main()
