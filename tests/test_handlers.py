async def test_doc_availability(client):
    resp = await client.get('/api/doc')

    result = await resp.text()

    assert "swagger" in result