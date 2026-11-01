def source_location(source):
    metadata = source.get('metadata', {})
    if metadata.get('source_type') == 'html':
        return 'веб-страница'
    suffix = ', OCR' if source['page'] in metadata.get('ocr_pages', []) else ''
    return f'стр. {source["page"]}{suffix}'


def source_link(source):
    url = source['source']
    if source.get('metadata', {}).get('source_type') == 'html':
        return url
    return url.split('#', 1)[0] + f'#page={source["page"]}'
